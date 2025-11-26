#!/usr/bin/env python3
"""
Uncertainty Quantification Analysis for Absolute Zero Reasoner

This script analyzes the correlation between uncertainty quantification methods
and the disagreement between ground truth (code execution) and majority vote labels.

Usage:
    python analysis/uncertainty_analysis.py \
        --model_path Qwen/Qwen2.5-Coder-3B \
        --seed_path data/3b_coder_seed_io.jsonl \
        --num_tasks 100 \
        --num_solver_samples 8 \
        --output_dir analysis/results
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from absolute_zero_reasoner.data_construction.prompts import (
    get_code_problem_generator_prompt,
    get_code_problem_predictor_prompt,
)
from absolute_zero_reasoner.data_construction.process_data import instruction_following
from absolute_zero_reasoner.rewards.code_reward import (
    parse_code_input_output,
    parse_inputs_message,
    parse_code_function,
)
from absolute_zero_reasoner.rewards.custom_evaluate import extract_answer
from absolute_zero_reasoner.utils.code_utils.python_executor import PythonExecutor


# Configuration from coder3b.sh
CONFIG = {
    "model_path": "andrewzh/Absolute_Zero_Reasoner-Coder-3b",
    "temperature": 1.0,
    "n_samples": 8,  # Number of solver samples for majority voting
    "problem_types": ["code_i", "code_o", "code_f"],
    "io_n": 6,  # Number of reference snippets
    "content_max_length": 5600,
    "max_prompt_length": 6144,
    "max_response_length": 8096,
    "num_inputs": 10,  # For code_f
    "banned_words": [
        "logging", "random", "multiprocessing", "pebble", "subprocess",
        "threading", "datetime", "time", "hashlib", "hmac", "bcrypt",
        "os.sys", "os.path", "sys.exit", "os.environ", "calendar"
    ],
    "banned_assertion_keywords": ["raise"],
    "extraction_type": "answer_conditional",
    "execute_max_timeout": 10,
    # vLLM hyperparameters from coder3b.sh
    "vllm": {
        "tensor_parallel_size": 2,
        "max_num_batched_tokens": 16384,
        "gpu_memory_utilization": 0.4,
        "enforce_eager": False,
        "dtype": "bfloat16",
    },
}


@dataclass
class TaskResult:
    """Results from evaluating a single task"""
    task_id: str
    problem_type: str
    snippet: str
    input_args: str
    ground_truth_output: str
    
    # Solver responses
    solver_responses: List[str] = field(default_factory=list)
    solver_extracted_answers: List[str] = field(default_factory=list)
    solver_execution_results: List[Optional[float]] = field(default_factory=list)
    
    # Majority vote
    majority_vote_answer: Optional[str] = None
    majority_vote_count: int = 0
    majority_vote_accuracy: float = 0.0
    
    # Ground truth from execution
    ground_truth_correct: List[bool] = field(default_factory=list)
    ground_truth_accuracy: float = 0.0
    
    # Disagreement
    gt_mv_disagree: bool = False  # Ground truth and majority vote disagree
    
    # Uncertainty metrics
    uncertainty_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass  
class UncertaintyMetrics:
    """Various uncertainty quantification metrics"""
    
    @staticmethod
    def entropy(probs: np.ndarray) -> float:
        """Shannon entropy of the distribution"""
        probs = probs[probs > 0]  # Avoid log(0)
        return -np.sum(probs * np.log2(probs))
    
    @staticmethod
    def normalized_entropy(probs: np.ndarray) -> float:
        """Normalized entropy (0 to 1)"""
        n_classes = len(probs)
        if n_classes <= 1:
            return 0.0
        max_entropy = np.log2(n_classes)
        return UncertaintyMetrics.entropy(probs) / max_entropy if max_entropy > 0 else 0.0
    
    @staticmethod
    def margin_of_victory(counts: Counter) -> float:
        """Difference between top 2 answers (normalized by total)"""
        if len(counts) == 0:
            return 0.0
        total = sum(counts.values())
        sorted_counts = sorted(counts.values(), reverse=True)
        if len(sorted_counts) == 1:
            return 1.0  # Complete agreement
        return (sorted_counts[0] - sorted_counts[1]) / total
    
    @staticmethod
    def agreement_ratio(counts: Counter) -> float:
        """Fraction of samples agreeing with majority"""
        if len(counts) == 0:
            return 0.0
        total = sum(counts.values())
        return max(counts.values()) / total
    
    @staticmethod
    def num_unique_answers(counts: Counter) -> int:
        """Number of unique answers"""
        return len(counts)
    
    @staticmethod
    def gini_impurity(probs: np.ndarray) -> float:
        """Gini impurity (1 - sum of squared probabilities)"""
        return 1 - np.sum(probs ** 2)
    
    @staticmethod
    def variance_of_correctness(execution_results: List[Optional[float]]) -> float:
        """Variance in execution accuracy across samples"""
        valid_results = [r for r in execution_results if r is not None]
        if len(valid_results) == 0:
            return 0.0
        return np.var(valid_results)
    
    @staticmethod
    def execution_failure_rate(execution_results: List[Optional[float]]) -> float:
        """Fraction of samples that failed execution"""
        n_total = len(execution_results)
        if n_total == 0:
            return 0.0
        n_failed = sum(1 for r in execution_results if r is None or r == 0.0)
        return n_failed / n_total


def load_seed_data(seed_path: str) -> List[Dict]:
    """Load seed IO data from JSONL file"""
    seed_data = []
    with open(seed_path, "r") as f:
        for line in f:
            seed_data.append(json.loads(line.strip()))
    return seed_data


def sample_references(seed_data: List[Dict], io_n: int) -> List[Dict]:
    """Sample reference snippets for task generation"""
    indices = np.random.choice(len(seed_data), size=min(io_n, len(seed_data)), replace=False)
    return [seed_data[i] for i in indices]


class ModelInference:
    """Wrapper for model inference using vLLM for fast generation"""
    
    def __init__(self, model_path: str, vllm_config: Dict = None):
        from vllm import LLM, SamplingParams
        
        print(f"Loading model from {model_path} using vLLM...")
        
        # Default vLLM config from coder3b.sh
        vllm_config = vllm_config or CONFIG.get("vllm", {})
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Apply base model chat template (from the codebase)
        self.tokenizer.chat_template = "{%- for message in messages -%}{{- '\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize vLLM with hyperparameters from coder3b.sh
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            tensor_parallel_size=vllm_config.get("tensor_parallel_size", 2),
            max_num_batched_tokens=vllm_config.get("max_num_batched_tokens", 16384),
            gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", 0.4),
            enforce_eager=vllm_config.get("enforce_eager", False),
            dtype=vllm_config.get("dtype", "bfloat16"),
            max_model_len=CONFIG["max_prompt_length"] + CONFIG["max_response_length"],
        )
        
        self.SamplingParams = SamplingParams
    
    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: int = 4096,
        num_samples: int = 1,
        do_sample: bool = True,
    ) -> List[str]:
        """Generate responses from the model using vLLM"""
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Set up sampling parameters
        sampling_params = self.SamplingParams(
            temperature=temperature if do_sample else 0.0,
            max_tokens=max_new_tokens,
            n=num_samples,
            top_p=1.0,
            top_k=-1,  # -1 for vllm means no top_k filtering
        )
        
        # Generate using vLLM
        outputs = self.llm.generate([text], sampling_params)
        
        # Extract generated text
        generated = []
        for output in outputs:
            for completion in output.outputs:
                generated.append(completion.text)
        
        return generated
    
    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        max_new_tokens: int = 4096,
        num_samples: int = 1,
        do_sample: bool = True,
    ) -> List[List[str]]:
        """Generate responses for multiple prompts in a batch (much faster)"""
        # Apply chat template to all prompts
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)
        
        # Set up sampling parameters
        sampling_params = self.SamplingParams(
            temperature=temperature if do_sample else 0.0,
            max_tokens=max_new_tokens,
            n=num_samples,
            top_p=1.0,
            top_k=-1,
        )
        
        # Generate using vLLM (batch processing)
        outputs = self.llm.generate(texts, sampling_params)
        
        # Extract generated text for each prompt
        all_generated = []
        for output in outputs:
            prompt_generated = []
            for completion in output.outputs:
                prompt_generated.append(completion.text)
            all_generated.append(prompt_generated)
        
        return all_generated


class TaskGenerator:
    """Generate tasks using the proposer/challenger role"""
    
    def __init__(
        self,
        model: ModelInference,
        executor: PythonExecutor,
        seed_data: List[Dict],
        config: Dict,
    ):
        self.model = model
        self.executor = executor
        self.seed_data = seed_data
        self.config = config
    
    def generate_task(self, problem_type: str) -> Optional[Dict]:
        """Generate a single task using the proposer"""
        # Sample references
        references = sample_references(self.seed_data, self.config["io_n"])
        
        # Build the proposer prompt
        raw_prompt = get_code_problem_generator_prompt(
            problem_type=problem_type,
            reference_snippets=references,
            banned_keywords=self.config["banned_words"],
            banned_assertion_keywords=self.config["banned_assertion_keywords"],
            composite_functions=[],
            remove_after_return=False,
            num_inputs=self.config["num_inputs"],
            remove_input_from_snippet=False,
        )
        
        # Apply instruction template
        prompt = instruction_following.format(raw_prompt)
        
        # Check prompt length
        if len(self.model.tokenizer(prompt)["input_ids"]) > self.config["content_max_length"]:
            return None
        
        # Generate task
        responses = self.model.generate(
            prompt,
            temperature=self.config["temperature"],
            max_new_tokens=2048,
            num_samples=1,
        )
        
        if not responses:
            return None
        
        response = responses[0]
        extracted = extract_answer(response, self.config["extraction_type"])
        
        # Parse the generated task
        if problem_type in ["code_i", "code_o"]:
            success, result = parse_code_input_output(
                extracted,
                parse_output=False,
                remove_after_return=False,
                remove_comments=False,
                remove_print=False,
                reject_multiple_functions=True,
                f_replace_location="not_first",
                reject_test_input_in_code=False,
                code_location="first",
            )
            
            if not success:
                return None
            
            # Validate and execute
            code_validity, output = self.executor.check_all(
                code=result["code"],
                inputs=result["input"],
                banned_keywords=self.config["banned_words"],
                check_determinism=True,
                imports=list(set(result["imports"])),
                check_error=False,
                banned_keywords_for_errors_and_exceptions=self.config["banned_assertion_keywords"],
            )
            
            if not code_validity:
                return None
            
            return {
                "problem_type": problem_type,
                "snippet": result["code"],
                "input": result["input"],
                "output": output,
                "imports": result["imports"],
            }
        
        elif problem_type == "code_f":
            success, result = parse_inputs_message(extracted, self.config["num_inputs"])
            
            if not success or len(result["inputs"]) != self.config["num_inputs"]:
                return None
            
            # Use first reference as the base snippet for code_f
            ref_snippet = references[0]["snippet"]
            ref_imports = references[0].get("imports", [])
            
            outputs = []
            for inp in result["inputs"]:
                code_validity, output = self.executor.check_all(
                    code=ref_snippet,
                    inputs=inp,
                    banned_keywords=[],
                    check_determinism=True,
                    imports=list(set(ref_imports)),
                    check_error=False,
                )
                if not code_validity:
                    return None
                outputs.append(output)
            
            return {
                "problem_type": problem_type,
                "snippet": ref_snippet,
                "inputs": result["inputs"],
                "outputs": outputs,
                "message": result["message"],
                "imports": ref_imports,
            }
        
        return None


class TaskSolver:
    """Solve tasks using the solver role"""
    
    def __init__(
        self,
        model: ModelInference,
        executor: PythonExecutor,
        config: Dict,
    ):
        self.model = model
        self.executor = executor
        self.config = config
    
    def solve_task(
        self,
        task: Dict,
        num_samples: int,
    ) -> TaskResult:
        """Solve a task and compute all metrics"""
        problem_type = task["problem_type"]
        
        # Build solver prompt
        if problem_type == "code_i":
            raw_prompt = get_code_problem_predictor_prompt(
                problem_type="code_i",
                snippet=task["snippet"],
                output=task["output"],
            )
        elif problem_type == "code_o":
            raw_prompt = get_code_problem_predictor_prompt(
                problem_type="code_o",
                snippet=task["snippet"],
                input_args=task["input"],
            )
        elif problem_type == "code_f":
            # Split inputs/outputs for code_f
            n_given = len(task["inputs"]) // 2
            given_inputs = task["inputs"][:n_given]
            given_outputs = task["outputs"][:n_given]
            hidden_inputs = task["inputs"][n_given:]
            hidden_outputs = task["outputs"][n_given:]
            
            raw_prompt = get_code_problem_predictor_prompt(
                problem_type="code_f",
                snippet=task["snippet"],
                message=task["message"],
                input_output_pairs=list(zip(given_inputs, given_outputs)),
            )
        
        prompt = instruction_following.format(raw_prompt)
        
        # Generate multiple solver samples using vLLM's native n parameter (much faster)
        responses = self.model.generate(
            prompt,
            temperature=self.config["temperature"],
            max_new_tokens=self.config["max_response_length"],
            num_samples=num_samples,
            do_sample=True,
        )
        
        # Process responses
        extracted_answers = []
        execution_results = []
        
        for resp in responses:
            extracted = extract_answer(resp, self.config["extraction_type"])
            
            if problem_type == "code_i":
                answer = self._extract_input(extracted)
                extracted_answers.append(answer)
                
                # Verify by execution
                if answer:
                    acc = self.executor.eval_input_prediction(
                        code=task["snippet"],
                        gold_output=task["output"],
                        agent_input=answer,
                        imports=task.get("imports", []),
                    )
                    execution_results.append(acc)
                else:
                    execution_results.append(0.0)
                    
            elif problem_type == "code_o":
                answer = self._extract_output(extracted)
                extracted_answers.append(answer)
                
                # Verify by execution
                if answer:
                    acc = self.executor.eval_output_prediction(
                        code=task["snippet"],
                        gold_output=task["output"],
                        agent_output=answer,
                        imports=task.get("imports", []),
                    )
                    execution_results.append(acc)
                else:
                    execution_results.append(0.0)
                    
            elif problem_type == "code_f":
                success, code = parse_code_function(extracted)
                extracted_answers.append(code if success else "")
                
                if success and code:
                    # Test on hidden inputs
                    accs = []
                    for inp, out in zip(hidden_inputs, hidden_outputs):
                        acc = self.executor.eval_input_prediction(
                            code=code,
                            gold_output=out,
                            agent_input=inp,
                            imports=task.get("imports", []),
                        )
                        if acc is not None:
                            accs.append(acc)
                    execution_results.append(np.mean(accs) if accs else 0.0)
                else:
                    execution_results.append(0.0)
        
        # Compute majority vote
        answer_counts = Counter(a for a in extracted_answers if a)
        if answer_counts:
            majority_answer, majority_count = answer_counts.most_common(1)[0]
        else:
            majority_answer, majority_count = None, 0
        
        # Compute ground truth accuracy
        valid_results = [r for r in execution_results if r is not None]
        gt_accuracy = np.mean([r for r in valid_results if r is not None]) if valid_results else 0.0
        
        # Check if majority vote matches ground truth criterion
        # Majority is "correct" if > 50% of execution results are correct
        mv_accuracy = majority_count / len(responses) if responses else 0.0
        
        # Determine if GT (execution) and MV disagree
        # GT says correct if execution accuracy > 0.5
        # MV says correct if majority answer appears in > 50% of samples
        gt_says_correct = gt_accuracy > 0.5
        mv_says_correct = mv_accuracy > 0.5
        gt_mv_disagree = gt_says_correct != mv_says_correct
        
        # Compute uncertainty metrics
        n_samples = len(responses)
        probs = np.array([c / n_samples for c in answer_counts.values()]) if answer_counts else np.array([])
        
        metrics = {
            "entropy": UncertaintyMetrics.entropy(probs) if len(probs) > 0 else 0.0,
            "normalized_entropy": UncertaintyMetrics.normalized_entropy(probs) if len(probs) > 0 else 0.0,
            "margin_of_victory": UncertaintyMetrics.margin_of_victory(answer_counts),
            "agreement_ratio": UncertaintyMetrics.agreement_ratio(answer_counts),
            "num_unique_answers": UncertaintyMetrics.num_unique_answers(answer_counts),
            "gini_impurity": UncertaintyMetrics.gini_impurity(probs) if len(probs) > 0 else 0.0,
            "variance_of_correctness": UncertaintyMetrics.variance_of_correctness(execution_results),
            "execution_failure_rate": UncertaintyMetrics.execution_failure_rate(execution_results),
        }
        
        return TaskResult(
            task_id=f"{problem_type}_{hash(task['snippet'][:100])}",
            problem_type=problem_type,
            snippet=task["snippet"],
            input_args=task.get("input", str(task.get("inputs", []))),
            ground_truth_output=task.get("output", str(task.get("outputs", []))),
            solver_responses=responses,
            solver_extracted_answers=extracted_answers,
            solver_execution_results=execution_results,
            majority_vote_answer=majority_answer,
            majority_vote_count=majority_count,
            majority_vote_accuracy=mv_accuracy,
            ground_truth_correct=[r == 1.0 for r in execution_results if r is not None],
            ground_truth_accuracy=gt_accuracy,
            gt_mv_disagree=gt_mv_disagree,
            uncertainty_metrics=metrics,
        )
    
    def _extract_input(self, content: str) -> str:
        """Extract input from solver response"""
        import re
        patterns = [
            r"```input\s*\n?(.*?)\n?```",
            r"# Input:\s*(.*?)(?=\n```|$)",
            r'input\s*\((.*?)\)',
            r"<input>\s*(.*?)(?:</input>|\s*$)",
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, content, re.DOTALL | re.IGNORECASE))
            if matches:
                return matches[-1].group(1).strip()
        
        return content.strip()
    
    def _extract_output(self, content: str) -> str:
        """Extract output from solver response"""
        import re
        patterns = [
            r"```output\s*\n?(.*?)\n?```",
            r"# Output:\s*(.*?)(?=\n```|$)",
            r'output\s*\((.*?)\)',
            r"<output>\s*(.*?)(?:</output>|\s*$)",
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, content, re.DOTALL | re.IGNORECASE))
            if matches:
                return matches[-1].group(1).strip()
        
        return content.strip()


def analyze_results(results: List[TaskResult], output_dir: str):
    """Analyze results and compute correlations"""
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    
    # Summary statistics
    n_total = len(results)
    n_disagree = sum(1 for r in results if r.gt_mv_disagree)
    
    print(f"\nTotal tasks analyzed: {n_total}")
    print(f"Tasks with GT-MV disagreement: {n_disagree} ({100*n_disagree/n_total:.1f}%)")
    
    # Collect metrics for analysis
    metrics_list = []
    for r in results:
        m = r.uncertainty_metrics.copy()
        m["gt_mv_disagree"] = int(r.gt_mv_disagree)
        m["problem_type"] = r.problem_type
        m["gt_accuracy"] = r.ground_truth_accuracy
        m["mv_accuracy"] = r.majority_vote_accuracy
        metrics_list.append(m)
    
    # Compute correlations between uncertainty metrics and disagreement
    print("\n" + "-" * 40)
    print("CORRELATION: Uncertainty Metrics vs GT-MV Disagreement")
    print("-" * 40)
    
    metric_names = [
        "entropy", "normalized_entropy", "margin_of_victory", 
        "agreement_ratio", "num_unique_answers", "gini_impurity",
        "variance_of_correctness", "execution_failure_rate"
    ]
    
    disagree_vals = np.array([m["gt_mv_disagree"] for m in metrics_list])
    
    correlations = {}
    for metric in metric_names:
        vals = np.array([m[metric] for m in metrics_list])
        if np.std(vals) > 0 and np.std(disagree_vals) > 0:
            corr = np.corrcoef(vals, disagree_vals)[0, 1]
        else:
            corr = 0.0
        correlations[metric] = corr
        
        # Higher is better for uncertainty detection
        direction = "higher = more uncertain" if metric not in ["margin_of_victory", "agreement_ratio"] else "lower = more uncertain"
        print(f"  {metric:30s}: r = {corr:+.3f} ({direction})")
    
    # Identify best predictor
    # For metrics where higher = more uncertain, positive correlation is good
    # For metrics where lower = more uncertain, negative correlation is good
    best_metric = max(
        correlations.items(),
        key=lambda x: abs(x[1])
    )
    print(f"\nBest predictor: {best_metric[0]} (|r| = {abs(best_metric[1]):.3f})")
    
    # Show examples of highly uncertain tasks
    print("\n" + "-" * 40)
    print("EXAMPLES: Highly Uncertain Tasks (with GT-MV Disagreement)")
    print("-" * 40)
    
    # Sort by normalized entropy (high uncertainty indicator)
    sorted_results = sorted(
        [r for r in results if r.gt_mv_disagree],
        key=lambda r: r.uncertainty_metrics["normalized_entropy"],
        reverse=True
    )[:5]  # Top 5
    
    for i, r in enumerate(sorted_results):
        print(f"\n[Example {i+1}]")
        print(f"  Problem type: {r.problem_type}")
        print(f"  Snippet: {r.snippet[:200]}...")
        print(f"  Ground truth accuracy: {r.ground_truth_accuracy:.2f}")
        print(f"  Majority vote accuracy: {r.majority_vote_accuracy:.2f}")
        print("  Uncertainty metrics:")
        for k, v in r.uncertainty_metrics.items():
            print(f"    {k}: {v:.3f}")
        print(f"  Number of unique answers: {r.uncertainty_metrics['num_unique_answers']}")
        if r.majority_vote_answer:
            print(f"  Majority vote answer: {r.majority_vote_answer[:100]}...")
    
    # Analyze diversity among highly uncertain samples
    print("\n" + "-" * 40)
    print("DIVERSITY ANALYSIS: Among Highly Uncertain Tasks")
    print("-" * 40)
    
    high_uncertainty_results = [
        r for r in results 
        if r.uncertainty_metrics["normalized_entropy"] > 0.5
    ]
    
    if high_uncertainty_results:
        # Count problem types
        type_counts = Counter(r.problem_type for r in high_uncertainty_results)
        print("\nDistribution of problem types among high-uncertainty tasks:")
        for pt, count in type_counts.most_common():
            print(f"  {pt}: {count}")
        
        # Unique snippets
        unique_snippets = len(set(r.snippet for r in high_uncertainty_results))
        print(f"\nUnique code snippets: {unique_snippets}/{len(high_uncertainty_results)}")
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    
    results_dict = {
        "summary": {
            "total_tasks": n_total,
            "disagree_count": n_disagree,
            "disagree_rate": n_disagree / n_total if n_total > 0 else 0,
        },
        "correlations": correlations,
        "best_predictor": {
            "metric": best_metric[0],
            "correlation": best_metric[1],
        },
        "tasks": [
            {
                "task_id": r.task_id,
                "problem_type": r.problem_type,
                "snippet": r.snippet,
                "input_args": r.input_args,
                "ground_truth_output": r.ground_truth_output,
                "majority_vote_answer": r.majority_vote_answer,
                "majority_vote_count": r.majority_vote_count,
                "majority_vote_accuracy": r.majority_vote_accuracy,
                "ground_truth_accuracy": r.ground_truth_accuracy,
                "gt_mv_disagree": r.gt_mv_disagree,
                "uncertainty_metrics": r.uncertainty_metrics,
                "solver_extracted_answers": r.solver_extracted_answers,
                "solver_execution_results": r.solver_execution_results,
            }
            for r in results
        ]
    }
    
    output_path = os.path.join(output_dir, "uncertainty_analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_path}")
    
    return correlations


def main():
    parser = argparse.ArgumentParser(description="Uncertainty Analysis for AZR")
    parser.add_argument("--model_path", type=str, default=CONFIG["model_path"])
    parser.add_argument("--seed_path", type=str, default="data/3b_coder_seed_io.jsonl")
    parser.add_argument("--num_tasks", type=int, default=100, help="Number of tasks to generate and analyze")
    parser.add_argument("--num_solver_samples", type=int, default=CONFIG["n_samples"], help="Number of solver samples per task")
    parser.add_argument("--output_dir", type=str, default="analysis/results")
    parser.add_argument("--problem_types", nargs="+", default=CONFIG["problem_types"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_generation", action="store_true", help="Skip task generation, use existing seed data directly")
    # vLLM configuration
    parser.add_argument("--tensor_parallel_size", type=int, default=CONFIG["vllm"]["tensor_parallel_size"],
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=CONFIG["vllm"]["gpu_memory_utilization"],
                       help="GPU memory utilization for vLLM")
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 80)
    print("UNCERTAINTY QUANTIFICATION ANALYSIS (using vLLM)")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Seed data: {args.seed_path}")
    print(f"  Number of tasks: {args.num_tasks}")
    print(f"  Solver samples per task: {args.num_solver_samples}")
    print(f"  Problem types: {args.problem_types}")
    print(f"  Tensor parallel size: {args.tensor_parallel_size}")
    print(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    
    # Load seed data
    print("\nLoading seed data...")
    seed_data = load_seed_data(args.seed_path)
    print(f"  Loaded {len(seed_data)} seed examples")
    
    # Update vLLM config with command line args
    vllm_config = CONFIG["vllm"].copy()
    vllm_config["tensor_parallel_size"] = args.tensor_parallel_size
    vllm_config["gpu_memory_utilization"] = args.gpu_memory_utilization
    
    # Initialize components
    print("\nInitializing vLLM model...")
    model = ModelInference(args.model_path, vllm_config=vllm_config)
    
    print("Initializing executor...")
    executor = PythonExecutor(
        timeout_length=CONFIG["execute_max_timeout"],
        ast_check=True,
        max_workers=1,
    )
    
    # Generate or use existing tasks
    if args.skip_generation:
        print("\nUsing seed data directly as tasks...")
        tasks = []
        for i, item in enumerate(seed_data[:args.num_tasks]):
            # Randomly assign problem type
            pt = np.random.choice(["code_i", "code_o"])
            tasks.append({
                "problem_type": pt,
                "snippet": item["snippet"],
                "input": item["input"],
                "output": item["output"],
                "imports": item.get("imports", []),
            })
    else:
        print("\nGenerating tasks using proposer...")
        generator = TaskGenerator(model, executor, seed_data, CONFIG)
        
        tasks = []
        tasks_per_type = args.num_tasks // len(args.problem_types)
        
        for problem_type in args.problem_types:
            print(f"  Generating {tasks_per_type} {problem_type} tasks...")
            pbar = tqdm(total=tasks_per_type, desc=f"Gen {problem_type}")
            attempts = 0
            max_attempts = tasks_per_type * 5
            
            while len([t for t in tasks if t["problem_type"] == problem_type]) < tasks_per_type and attempts < max_attempts:
                task = generator.generate_task(problem_type)
                if task:
                    tasks.append(task)
                    pbar.update(1)
                attempts += 1
            
            pbar.close()
    
    print(f"\nGenerated {len(tasks)} valid tasks")
    
    # Solve tasks and collect results
    print("\nSolving tasks and computing metrics...")
    solver = TaskSolver(model, executor, CONFIG)
    
    results = []
    for task in tqdm(tasks, desc="Solving"):
        try:
            result = solver.solve_task(task, args.num_solver_samples)
            results.append(result)
        except Exception as e:
            print(f"Error solving task: {e}")
            continue
    
    print(f"\nSuccessfully solved {len(results)} tasks")
    
    # Analyze results
    analyze_results(results, args.output_dir)
    
    # Cleanup
    executor.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()

