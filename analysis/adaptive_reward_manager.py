"""
Adaptive Reward Manager for Uncertainty-Guided Verification

This module extends the CodeIORewardManager to support three verification modes:
1. FULL_EXECUTION: Always use Python executor (ground truth)
2. FULL_LLM: Always use LLM-as-a-judge
3. ADAPTIVE: Use uncertainty to decide, with a budget constraint

The adaptive mode uses uncertainty quantification to determine when the LLM
verification is likely to be unreliable, and falls back to execution.
"""

import os
import sys
import time
import math
from typing import List, Dict, Tuple, Optional, Literal
from collections import Counter
from enum import Enum
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm import LLM, SamplingParams

from absolute_zero_reasoner.rewards.reward_managers import CodeIORewardManager
from absolute_zero_reasoner.rewards.code_reward import parse_code_input_output
from absolute_zero_reasoner.utils.code_utils.python_executor import PythonExecutor


class VerificationMode(Enum):
    """Verification strategy modes."""
    FULL_EXECUTION = "full_execution"
    FULL_LLM = "full_llm"
    ADAPTIVE = "adaptive"


class AdaptiveRewardManager:
    """
    Reward manager that supports different verification strategies.

    This is a simplified version focused on prediction tasks (code_i, code_o)
    for the initial training step analysis.
    """

    def __init__(
        self,
        executor: PythonExecutor,
        llm: LLM,
        mode: VerificationMode = VerificationMode.FULL_EXECUTION,
        budget_fraction: float = 0.3,
        uncertainty_threshold: float = 0.5,
        n_samples_for_uq: int = 8,
        config: Dict = None,
    ):
        """
        Args:
            executor: PythonExecutor for ground truth verification
            llm: VLLM LLM instance for LLM-based verification
            mode: Verification mode (FULL_EXECUTION, FULL_LLM, ADAPTIVE)
            budget_fraction: Fraction of tasks to verify with execution in ADAPTIVE mode
            uncertainty_threshold: Threshold above which to use execution in ADAPTIVE mode
            n_samples_for_uq: Number of samples for uncertainty quantification
            config: Additional config options
        """
        self.executor = executor
        self.llm = llm
        self.mode = mode
        self.budget_fraction = budget_fraction
        self.uncertainty_threshold = uncertainty_threshold
        self.n_samples_for_uq = n_samples_for_uq
        self.config = config or {}

        # Tracking metrics
        self.stats = {
            "total_verifications": 0,
            "execution_verifications": 0,
            "llm_verifications": 0,
            "execution_time_ms": 0.0,
            "llm_time_ms": 0.0,
            "label_agreements": 0,
            "label_disagreements": 0,
        }

    def reset_stats(self):
        """Reset tracking statistics."""
        for key in self.stats:
            self.stats[key] = 0 if isinstance(self.stats[key], int) else 0.0

    def compute_rewards(
        self,
        tasks: List[Dict],
        problem_type: str,
    ) -> List[Dict]:
        """
        Compute rewards for a batch of tasks using the configured verification mode.

        Args:
            tasks: List of task dicts with solver responses
            problem_type: "code_i" or "code_o"

        Returns:
            List of task dicts with rewards and verification results
        """
        if self.mode == VerificationMode.FULL_EXECUTION:
            return self._verify_all_execution(tasks, problem_type)
        elif self.mode == VerificationMode.FULL_LLM:
            return self._verify_all_llm(tasks, problem_type)
        elif self.mode == VerificationMode.ADAPTIVE:
            return self._verify_adaptive(tasks, problem_type)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _verify_all_execution(
        self,
        tasks: List[Dict],
        problem_type: str,
    ) -> List[Dict]:
        """Verify all tasks using Python execution."""
        results = []

        for task in tasks:
            start_time = time.time()

            correct, exec_result = self._verify_with_execution(
                task, problem_type
            )

            exec_time = (time.time() - start_time) * 1000

            task_result = {
                **task,
                "reward": 1.0 if correct else -0.5,
                "verification_correct": correct,
                "verification_method": "execution",
                "verification_time_ms": exec_time,
                "execution_result": exec_result,
            }
            results.append(task_result)

            self.stats["total_verifications"] += 1
            self.stats["execution_verifications"] += 1
            self.stats["execution_time_ms"] += exec_time

        return results

    def _verify_all_llm(
        self,
        tasks: List[Dict],
        problem_type: str,
    ) -> List[Dict]:
        """Verify all tasks using LLM-as-a-judge."""
        # Batch LLM verification for efficiency
        verifier_prompts = []
        valid_indices = []

        for idx, task in enumerate(tasks):
            if task.get("solver_answer") is None:
                continue

            prompt = self._construct_verifier_prompt(task, problem_type)
            verifier_prompts.append(prompt)
            valid_indices.append(idx)

        # Generate LLM verdicts
        start_time = time.time()

        if verifier_prompts:
            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=512,
                n=1,
            )
            outputs = self.llm.generate(verifier_prompts, sampling_params)
        else:
            outputs = []

        llm_time = (time.time() - start_time) * 1000
        time_per_task = llm_time / len(verifier_prompts) if verifier_prompts else 0

        # Process results
        results = []
        llm_results = {}

        for out_idx, output in enumerate(outputs):
            task_idx = valid_indices[out_idx]
            response = output.outputs[0].text
            verdict = self._extract_llm_verdict(response)
            llm_results[task_idx] = {
                "verdict": verdict,
                "response": response,
            }

        for idx, task in enumerate(tasks):
            if idx in llm_results:
                correct = llm_results[idx]["verdict"]
                task_result = {
                    **task,
                    "reward": 1.0 if correct else -0.5,
                    "verification_correct": correct,
                    "verification_method": "llm",
                    "verification_time_ms": time_per_task,
                    "llm_response": llm_results[idx]["response"],
                }
            else:
                task_result = {
                    **task,
                    "reward": -1.0,  # Invalid task
                    "verification_correct": None,
                    "verification_method": "none",
                    "verification_time_ms": 0,
                }

            results.append(task_result)

            self.stats["total_verifications"] += 1
            self.stats["llm_verifications"] += 1
            self.stats["llm_time_ms"] += time_per_task

        return results

    def _verify_adaptive(
        self,
        tasks: List[Dict],
        problem_type: str,
    ) -> List[Dict]:
        """
        Verify tasks adaptively based on uncertainty.

        Strategy:
        1. Compute uncertainty for each task using self-consistency
        2. Sort by uncertainty (highest first)
        3. Use execution for top budget_fraction tasks
        4. Use LLM for the rest
        """
        # First, compute uncertainty for all tasks
        tasks_with_uq = self._compute_uncertainties(tasks, problem_type)

        # Sort by uncertainty (descending)
        sorted_indices = sorted(
            range(len(tasks_with_uq)),
            key=lambda i: tasks_with_uq[i].get("uncertainty", 0),
            reverse=True
        )

        # Determine budget
        n_execution = int(len(tasks) * self.budget_fraction)
        execution_indices = set(sorted_indices[:n_execution])
        llm_indices = set(sorted_indices[n_execution:])

        # Verify with execution (high uncertainty)
        execution_tasks = [tasks_with_uq[i] for i in execution_indices]
        llm_tasks_indices = [(i, tasks_with_uq[i]) for i in llm_indices]

        results = [None] * len(tasks)

        # Execution verification
        for i in execution_indices:
            task = tasks_with_uq[i]
            start_time = time.time()

            correct, exec_result = self._verify_with_execution(task, problem_type)
            exec_time = (time.time() - start_time) * 1000

            results[i] = {
                **task,
                "reward": 1.0 if correct else -0.5,
                "verification_correct": correct,
                "verification_method": "execution",
                "verification_time_ms": exec_time,
                "execution_result": exec_result,
                "selected_for_execution": True,
            }

            self.stats["execution_verifications"] += 1
            self.stats["execution_time_ms"] += exec_time

        # LLM verification (batch)
        llm_prompts = []
        llm_task_map = []

        for i in llm_indices:
            task = tasks_with_uq[i]
            if task.get("solver_answer") is not None:
                prompt = self._construct_verifier_prompt(task, problem_type)
                llm_prompts.append(prompt)
                llm_task_map.append(i)

        if llm_prompts:
            start_time = time.time()
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=512, n=1
            )
            outputs = self.llm.generate(llm_prompts, sampling_params)
            llm_time = (time.time() - start_time) * 1000
            time_per_task = llm_time / len(llm_prompts)

            for out_idx, output in enumerate(outputs):
                task_idx = llm_task_map[out_idx]
                task = tasks_with_uq[task_idx]
                response = output.outputs[0].text
                verdict = self._extract_llm_verdict(response)

                results[task_idx] = {
                    **task,
                    "reward": 1.0 if verdict else -0.5,
                    "verification_correct": verdict,
                    "verification_method": "llm",
                    "verification_time_ms": time_per_task,
                    "llm_response": response,
                    "selected_for_execution": False,
                }

                self.stats["llm_verifications"] += 1
                self.stats["llm_time_ms"] += time_per_task

        # Handle any remaining None entries
        for i, r in enumerate(results):
            if r is None:
                results[i] = {
                    **tasks_with_uq[i],
                    "reward": -1.0,
                    "verification_correct": None,
                    "verification_method": "none",
                    "verification_time_ms": 0,
                    "selected_for_execution": False,
                }

        self.stats["total_verifications"] += len(tasks)

        return results

    def _compute_uncertainties(
        self,
        tasks: List[Dict],
        problem_type: str,
    ) -> List[Dict]:
        """
        Compute uncertainty for each task using self-consistency.

        Generates multiple solver responses and measures agreement.
        """
        if self.n_samples_for_uq <= 1:
            # No UQ, assign default uncertainty
            for task in tasks:
                task["uncertainty"] = 0.5
            return tasks

        # Generate multiple solver responses
        solver_prompts = []
        valid_indices = []

        for idx, task in enumerate(tasks):
            prompt = self._construct_solver_prompt(task, problem_type)
            if prompt:
                solver_prompts.append(prompt)
                valid_indices.append(idx)

        if not solver_prompts:
            for task in tasks:
                task["uncertainty"] = 0.5
            return tasks

        # Generate with multiple samples
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024,
            n=self.n_samples_for_uq,
        )

        outputs = self.llm.generate(solver_prompts, sampling_params)

        # Compute uncertainty for each task
        for out_idx, output in enumerate(outputs):
            task_idx = valid_indices[out_idx]
            task = tasks[task_idx]

            # Extract all answers
            answers = []
            for sample in output.outputs:
                answer = self._extract_answer(sample.text, problem_type)
                answers.append(answer)

            # Store first valid answer as primary
            valid_answers = [a for a in answers if a is not None]
            if valid_answers:
                task["solver_answer"] = valid_answers[0]

            # Compute self-consistency entropy
            uncertainty = self._compute_self_consistency_entropy(answers)
            task["uncertainty"] = uncertainty
            task["n_samples"] = len(answers)
            task["unique_answers"] = len(set(str(a) for a in answers if a is not None))

        # Assign default uncertainty to tasks without valid prompts
        for idx, task in enumerate(tasks):
            if "uncertainty" not in task:
                task["uncertainty"] = 0.5

        return tasks

    def _compute_self_consistency_entropy(self, answers: List) -> float:
        """Compute normalized entropy of answer distribution."""
        # Count answer frequencies
        answer_strs = [str(a) if a is not None else "__NONE__" for a in answers]
        counts = Counter(answer_strs)

        n = len(answer_strs)
        if n == 0:
            return 1.0

        # Compute entropy
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by max entropy
        max_entropy = math.log2(n) if n > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return normalized_entropy

    def _verify_with_execution(
        self,
        task: Dict,
        problem_type: str,
    ) -> Tuple[bool, str]:
        """Verify using Python execution."""
        code = task.get("code_snippet", "")
        predicted = task.get("solver_answer", "")
        gold = task.get("gold_output", "")
        imports = task.get("imports", [])

        if not code or predicted is None:
            return False, "missing_data"

        try:
            if problem_type == "code_o":
                accuracy = self.executor.eval_output_prediction(
                    code=code,
                    gold_output=gold,
                    agent_output=str(predicted),
                    imports=imports,
                )
                return accuracy == 1.0, f"accuracy={accuracy}"

            elif problem_type == "code_i":
                accuracy = self.executor.eval_input_prediction(
                    code=code,
                    gold_output=gold,
                    agent_input=str(predicted),
                    imports=imports,
                )
                return accuracy == 1.0, f"accuracy={accuracy}"

            return False, "unknown_problem_type"

        except Exception as e:
            return False, f"execution_error: {str(e)}"

    def _construct_solver_prompt(self, task: Dict, problem_type: str) -> Optional[str]:
        """Construct solver prompt for a task."""
        code = task.get("code_snippet", "")
        if not code:
            return None

        if problem_type == "code_o":
            input_args = task.get("input_args", "")
            return f"""Given the following Python code and input, predict the output.

```python
{code}
```

Input: {input_args}

What is the output? Provide only the output value."""

        elif problem_type == "code_i":
            gold_output = task.get("gold_output", "")
            return f"""Given the following Python code and expected output, predict the input.

```python
{code}
```

Expected Output: {gold_output}

What input would produce this output? Provide only the input value."""

        return None

    def _construct_verifier_prompt(self, task: Dict, problem_type: str) -> str:
        """Construct LLM verifier prompt."""
        code = task.get("code_snippet", "")
        predicted = task.get("solver_answer", "")
        gold = task.get("gold_output", "")
        input_args = task.get("input_args", "")

        if problem_type == "code_o":
            return f"""You are a code verification assistant. Determine if the predicted output is correct.

Code:
```python
{code}
```

Input: {input_args}
Expected Output: {gold}
Predicted Output: {predicted}

Is the predicted output correct? Answer with only "CORRECT" or "INCORRECT"."""

        elif problem_type == "code_i":
            return f"""You are a code verification assistant. Determine if the predicted input would produce the expected output.

Code:
```python
{code}
```

Expected Output: {gold}
Predicted Input: {predicted}

Would this input produce the expected output? Answer with only "CORRECT" or "INCORRECT"."""

        return ""

    def _extract_answer(self, response: str, problem_type: str) -> Optional[str]:
        """Extract answer from solver response."""
        response = response.strip()

        # Try to extract from code block
        if "```" in response:
            parts = response.split("```")
            for part in parts[1::2]:  # Odd indices are code blocks
                lines = part.strip().split("\n")
                if lines and not lines[0].startswith("python"):
                    return lines[0].strip()
                elif len(lines) > 1:
                    return lines[1].strip()

        # Take first line as answer
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                return line

        return response if response else None

    def _extract_llm_verdict(self, response: str) -> bool:
        """Extract verdict from LLM verifier response."""
        response_upper = response.upper()

        if "CORRECT" in response_upper and "INCORRECT" not in response_upper:
            return True
        if "INCORRECT" in response_upper:
            return False
        if "YES" in response_upper:
            return True
        if "NO" in response_upper:
            return False

        # Default to incorrect if unclear
        return False

    def get_stats_summary(self) -> Dict:
        """Get summary of verification statistics."""
        total = self.stats["total_verifications"]
        if total == 0:
            return self.stats

        summary = {
            **self.stats,
            "execution_fraction": self.stats["execution_verifications"] / total,
            "llm_fraction": self.stats["llm_verifications"] / total,
            "avg_execution_time_ms": (
                self.stats["execution_time_ms"] / self.stats["execution_verifications"]
                if self.stats["execution_verifications"] > 0 else 0
            ),
            "avg_llm_time_ms": (
                self.stats["llm_time_ms"] / self.stats["llm_verifications"]
                if self.stats["llm_verifications"] > 0 else 0
            ),
        }

        return summary


def create_reward_manager(
    mode: str,
    executor: PythonExecutor,
    llm: LLM,
    budget_fraction: float = 0.3,
    uncertainty_threshold: float = 0.5,
    n_samples_for_uq: int = 8,
    **kwargs,
) -> AdaptiveRewardManager:
    """
    Factory function to create reward manager with specified mode.

    Args:
        mode: "full_execution", "full_llm", or "adaptive"
        executor: PythonExecutor instance
        llm: VLLM LLM instance
        budget_fraction: For adaptive mode, fraction of tasks using execution
        uncertainty_threshold: For adaptive mode, uncertainty threshold
        n_samples_for_uq: Number of samples for uncertainty estimation

    Returns:
        AdaptiveRewardManager instance
    """
    mode_enum = VerificationMode(mode)

    return AdaptiveRewardManager(
        executor=executor,
        llm=llm,
        mode=mode_enum,
        budget_fraction=budget_fraction,
        uncertainty_threshold=uncertainty_threshold,
        n_samples_for_uq=n_samples_for_uq,
        config=kwargs,
    )
