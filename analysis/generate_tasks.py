"""
Task Generation and Solving Analysis Script for Absolute Zero Reasoner

This script generates tasks using the challenger/proposer from a pretrained model
and evaluates solver performance using both code execution and LLM-based verification.

Two labels are produced for each task:
1. Execution label: Ground truth from actually running the Python code
2. LLM label: The LLM judges if the solver's answer matches the gold answer (no code execution)

Additionally, multiple uncertainty quantification (UQ) methods are computed.

Usage:
    # Generate tasks only
    python -m analysis.generate_tasks --num_tasks 10

    # Generate and solve tasks with both labels
    python -m analysis.generate_tasks --num_tasks 5 --solve

    # With uncertainty quantification (requires multiple samples)
    python -m analysis.generate_tasks --num_tasks 5 --solve --n_samples 8

    # Show prompts and save results
    python -m analysis.generate_tasks --num_tasks 5 --solve --n_samples 8 --show_prompts --output_path results.json
"""

import os
import sys
import json
import argparse
import re
import math
import time
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np
from numpy import random

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from absolute_zero_reasoner.data_construction.prompts import (
    get_code_problem_generator_prompt,
    get_code_problem_predictor_prompt,
)
from absolute_zero_reasoner.data_construction.process_data import instruction_following
from absolute_zero_reasoner.rewards.code_reward import parse_code_input_output
from absolute_zero_reasoner.utils.code_utils.python_executor import PythonExecutor


# Hyperparameters from coder3b.sh
CONFIG = {
    # Model config
    "model_path": "andrewzh/Absolute_Zero_Reasoner-Coder-3b",
    "base_model": "Qwen/Qwen2.5-Coder-3B",  # For tokenizer if needed

    # Seed data paths
    "seed_dataset": "data/3b_coder_seed_io.jsonl",
    "code_f_seed_dataset": "data/3b_coder_code_f_seed_io.jsonl",

    # Data construction config (from azr_ppo_trainer.yaml and coder3b.sh)
    "io_n": 6,  # Number of reference snippets
    "content_max_length": 5600,  # Max content length in tokens
    "num_inputs": 10,  # For code_f task

    # VLLM config (from coder3b.sh)
    "tensor_parallel_size": 1,  # Use 1 for single GPU analysis
    "gpu_memory_utilization": 0.4,
    "max_num_batched_tokens": 16384,
    "dtype": "bfloat16",
    "enforce_eager": False,

    # Generation config
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 8096,  # max_response_length from config

    # Uncertainty quantification config
    "n_samples": 8,  # Number of samples for UQ methods

    # Executor config
    "execute_max_timeout": 10,  # seconds
    "ast_check": True,

    # Banned keywords (from config)
    "banned_keywords": [
        "logging", "random", "multiprocessing", "pebble", "subprocess",
        "threading", "datetime", "time", "hashlib", "hmac", "bcrypt",
        "os.sys", "os.path", "sys.exit", "os.environ", "calendar"
    ],
    "banned_assertion_keywords": [],  # Empty for non-error tasks

    # Problem types
    "problem_types": ["code_i", "code_o", "code_f"],
}


# Verification prompt templates for LLM-based answer comparison
VERIFY_OUTPUT_PROMPT = """You are a Python code verification assistant. Your task is to determine if two values are equivalent.

Given the following code snippet:
```python
{code_snippet}
```

The code was executed with input: {input_args}

Expected output (gold answer): {gold_answer}
Predicted output (solver's answer): {predicted_answer}

Are these two outputs equivalent? Consider that:
- Different string representations of the same value should be considered equivalent (e.g., "1" and 1, or {{"a": 1}} and {{"a": 1}})
- Floating point numbers should be compared with reasonable tolerance
- Order of elements in sets or dictionary keys may differ but values should match

Think step by step, then provide your final verdict.

Your response MUST end with exactly one of these lines:
```verdict
CORRECT
```
or
```verdict
INCORRECT
```
"""

VERIFY_INPUT_PROMPT = """You are a Python code verification assistant. Your task is to determine if a predicted input would produce the expected output.

Given the following code snippet:
```python
{code_snippet}
```

Expected output (gold answer): {gold_answer}
Predicted input (solver's answer): {predicted_answer}

Would the predicted input, when passed to function f(), produce the expected output?

Think step by step about what the function does and whether the predicted input would result in the gold output.

Your response MUST end with exactly one of these lines:
```verdict
CORRECT
```
or
```verdict
INCORRECT
```
"""

# Verbalized confidence prompt
VERBALIZED_CONFIDENCE_PROMPT = """You just solved the following problem:

{problem_description}

Your answer was: {answer}

On a scale from 0 to 100, how confident are you that your answer is correct?
- 0 means completely uncertain (random guess)
- 100 means absolutely certain

Respond with ONLY a number between 0 and 100, nothing else.
"""


# ==============================================================================
# UNCERTAINTY QUANTIFICATION METHODS
# ==============================================================================

def compute_entropy(probs: List[float]) -> float:
    """Compute Shannon entropy from a probability distribution."""
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_self_consistency_entropy(answers: List[str]) -> Dict[str, float]:
    """
    Compute Self-Consistency Entropy from multiple sampled answers.

    **Why it quantifies uncertainty:**
    Self-consistency is based on the principle that a confident model will produce
    consistent answers across multiple samples. If the model is uncertain, it will
    "explore" different answers due to the stochasticity in sampling. High entropy
    in the answer distribution indicates the model is spread across multiple possible
    answers, signaling high uncertainty.

    Returns:
        - entropy: Shannon entropy of the answer distribution (higher = more uncertain)
        - normalized_entropy: Entropy normalized by log2(n_unique) for comparability
    """
    if not answers:
        return {"sc_entropy": float('nan'), "sc_entropy_normalized": float('nan')}

    # Filter out None answers
    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        return {"sc_entropy": float('nan'), "sc_entropy_normalized": float('nan')}

    # Count answer frequencies
    counter = Counter(valid_answers)
    total = len(valid_answers)

    # Compute probabilities
    probs = [count / total for count in counter.values()]

    # Compute entropy
    entropy = compute_entropy(probs)

    # Normalized entropy (0 to 1 scale)
    n_unique = len(counter)
    max_entropy = math.log2(n_unique) if n_unique > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "sc_entropy": entropy,
        "sc_entropy_normalized": normalized_entropy,
    }


def compute_agreement_rate(answers: List[str]) -> Dict[str, float]:
    """
    Compute Agreement Rate - the fraction of samples agreeing with the majority answer.

    **Why it quantifies uncertainty:**
    Agreement rate directly measures consensus among multiple samples. A high agreement
    rate means the model consistently produces the same answer, indicating confidence.
    A low agreement rate means samples disagree, indicating the model is uncertain about
    which answer is correct. This is more interpretable than entropy.

    Returns:
        - agreement_rate: Fraction of samples matching the majority (0 to 1, higher = more certain)
        - n_unique_answers: Number of distinct answers (more = more uncertain)
    """
    if not answers:
        return {"agreement_rate": float('nan'), "n_unique_answers": 0}

    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        return {"agreement_rate": float('nan'), "n_unique_answers": 0}

    counter = Counter(valid_answers)
    most_common_count = counter.most_common(1)[0][1]
    agreement_rate = most_common_count / len(valid_answers)

    return {
        "agreement_rate": agreement_rate,
        "n_unique_answers": len(counter),
    }


def compute_sequence_logprob(cumulative_logprob: float, n_tokens: int) -> Dict[str, float]:
    """
    Compute sequence-level log probability metrics.

    **Why it quantifies uncertainty:**
    The model assigns log probabilities to each generated token based on its confidence.
    A high (less negative) log probability means the model is confident in its generation.
    A low (more negative) log probability means the model had to choose among many
    plausible continuations, indicating uncertainty. This captures token-level uncertainty
    aggregated over the whole sequence.

    Returns:
        - seq_logprob: Total log probability of the sequence (higher = more certain)
        - seq_logprob_per_token: Average log probability per token (normalized)
        - perplexity: Exp of negative average log prob (lower = more certain)
    """
    if n_tokens == 0:
        return {
            "seq_logprob": float('nan'),
            "seq_logprob_per_token": float('nan'),
            "perplexity": float('nan'),
        }

    avg_logprob = cumulative_logprob / n_tokens
    perplexity = math.exp(-avg_logprob)

    return {
        "seq_logprob": cumulative_logprob,
        "seq_logprob_per_token": avg_logprob,
        "perplexity": perplexity,
    }


def compute_lexical_diversity(answers: List[str]) -> Dict[str, float]:
    """
    Compute lexical diversity among multiple sampled answers.

    **Why it quantifies uncertainty:**
    When a model is uncertain, different samples may use different words, phrasings,
    or structures to express answers. High lexical diversity indicates the model is
    exploring different ways to respond, which correlates with uncertainty. This
    complements semantic measures by capturing surface-level variation.

    Uses:
    - Type-Token Ratio (TTR): Ratio of unique tokens to total tokens
    - Pairwise edit distance: Average normalized edit distance between answer pairs

    Returns:
        - lexical_ttr: Type-token ratio across all answers (higher = more diverse)
        - avg_pairwise_edit_dist: Average normalized edit distance (higher = more diverse)
    """
    if not answers:
        return {"lexical_ttr": float('nan'), "avg_pairwise_edit_dist": float('nan')}

    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        return {"lexical_ttr": float('nan'), "avg_pairwise_edit_dist": float('nan')}

    # Type-Token Ratio
    all_tokens = []
    for ans in valid_answers:
        all_tokens.extend(ans.split())

    if all_tokens:
        ttr = len(set(all_tokens)) / len(all_tokens)
    else:
        ttr = 0.0

    # Pairwise edit distance (Levenshtein-like, normalized)
    def normalized_edit_distance(s1: str, s2: str) -> float:
        """Compute normalized edit distance between two strings."""
        if not s1 and not s2:
            return 0.0
        if not s1 or not s2:
            return 1.0

        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n] / max(m, n)

    # Compute pairwise distances
    distances = []
    for i in range(len(valid_answers)):
        for j in range(i + 1, len(valid_answers)):
            dist = normalized_edit_distance(valid_answers[i], valid_answers[j])
            distances.append(dist)

    avg_edit_dist = sum(distances) / len(distances) if distances else 0.0

    return {
        "lexical_ttr": ttr,
        "avg_pairwise_edit_dist": avg_edit_dist,
    }


def compute_answer_variance(answers: List[str]) -> Dict[str, float]:
    """
    Compute variance for numerical answers.

    **Why it quantifies uncertainty:**
    For tasks with numerical outputs, variance directly measures the spread of
    predictions. High variance means the model produces widely different numerical
    values across samples, indicating uncertainty about the correct value. This is
    especially useful for regression-like tasks or when answers are numbers.

    Returns:
        - numeric_variance: Variance of numerical answers (nan if not numeric)
        - numeric_std: Standard deviation
        - numeric_range: Range (max - min) of values
    """
    if not answers:
        return {"numeric_variance": float('nan'), "numeric_std": float('nan'), "numeric_range": float('nan')}

    # Try to parse answers as numbers
    numeric_values = []
    for ans in answers:
        if ans is None:
            continue
        try:
            val = float(ans.strip())
            numeric_values.append(val)
        except (ValueError, AttributeError):
            # Try to extract number from string
            match = re.search(r'-?\d+\.?\d*', str(ans))
            if match:
                try:
                    val = float(match.group())
                    numeric_values.append(val)
                except ValueError:
                    pass

    if len(numeric_values) < 2:
        return {"numeric_variance": float('nan'), "numeric_std": float('nan'), "numeric_range": float('nan')}

    variance = np.var(numeric_values)
    std = np.std(numeric_values)
    value_range = max(numeric_values) - min(numeric_values)

    return {
        "numeric_variance": float(variance),
        "numeric_std": float(std),
        "numeric_range": float(value_range),
    }


def extract_verbalized_confidence(response: str) -> Optional[float]:
    """Extract confidence score (0-100) from verbalized confidence response."""
    # Try to find a number in the response
    match = re.search(r'\b(\d{1,3})\b', response.strip())
    if match:
        conf = int(match.group(1))
        if 0 <= conf <= 100:
            return conf / 100.0  # Normalize to 0-1
    return None


def compute_all_uncertainty_metrics(
    answers: List[str],
    logprobs: List[Tuple[float, int]] = None,  # List of (cumulative_logprob, n_tokens)
) -> Dict[str, float]:
    """
    Compute all uncertainty quantification metrics.

    Args:
        answers: List of extracted answers from multiple samples
        logprobs: List of (cumulative_logprob, n_tokens) for each sample

    Returns:
        Dictionary with all UQ metrics
    """
    metrics = {}

    # 1. Self-Consistency Entropy
    sc_metrics = compute_self_consistency_entropy(answers)
    metrics.update(sc_metrics)

    # 2. Agreement Rate
    agreement_metrics = compute_agreement_rate(answers)
    metrics.update(agreement_metrics)

    # 3. Lexical Diversity
    diversity_metrics = compute_lexical_diversity(answers)
    metrics.update(diversity_metrics)

    # 4. Numeric Variance (if applicable)
    variance_metrics = compute_answer_variance(answers)
    metrics.update(variance_metrics)

    # 5. Sequence Log Probability (if available)
    if logprobs:
        # Average across samples
        avg_logprob = sum(lp for lp, _ in logprobs) / len(logprobs)
        avg_tokens = sum(nt for _, nt in logprobs) / len(logprobs)
        logprob_metrics = compute_sequence_logprob(avg_logprob, avg_tokens)
        metrics.update(logprob_metrics)

        # Also compute variance of log probs across samples
        if len(logprobs) > 1:
            per_token_logprobs = [lp/nt if nt > 0 else 0 for lp, nt in logprobs]
            metrics["logprob_variance"] = float(np.var(per_token_logprobs))

    return metrics


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def load_seed_data(seed_path: str) -> List[Dict]:
    """Load seed data from jsonl file."""
    data = []
    with open(seed_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def construct_proposer_prompt(
    problem_type: str,
    seed_data: List[Dict],
    config: Dict,
) -> Tuple[str, List[Dict]]:
    """
    Construct a proposer prompt for task generation.
    Uses the same logic as get_gen_code_io_data in constructor.py.
    """
    # Sample reference snippets
    if problem_type == 'code_f':
        chosen_ref = random.choice(seed_data, size=1, replace=False)[0]
        reference_snippets = [{
            'snippet': chosen_ref['snippet'],
            'input': chosen_ref.get('inputs', [''])[0] if chosen_ref.get('inputs') else '',
            'output': chosen_ref.get('outputs', [''])[0] if chosen_ref.get('outputs') else '',
            'imports': chosen_ref.get('imports', []),
        }]
    else:
        chosen_references = random.choice(
            seed_data,
            size=min(config["io_n"], len(seed_data)),
            replace=False
        ).tolist()
        reference_snippets = chosen_references

    generator_prompt = get_code_problem_generator_prompt(
        problem_type=problem_type,
        reference_snippets=reference_snippets,
        banned_keywords=config["banned_keywords"],
        banned_assertion_keywords=config["banned_assertion_keywords"],
        composite_functions=[],
        remove_after_return=False,
        num_inputs=config["num_inputs"],
        remove_input_from_snippet=False,
    )

    full_prompt = instruction_following.format(generator_prompt)
    return full_prompt, reference_snippets


def construct_solver_prompt(
    problem_type: str,
    code_snippet: str,
    input_args: str = None,
    output: str = None,
) -> str:
    """Construct a solver prompt for a given task."""
    solver_prompt = get_code_problem_predictor_prompt(
        problem_type=problem_type,
        snippet=code_snippet,
        input_args=input_args,
        output=output,
    )
    full_prompt = instruction_following.format(solver_prompt)
    return full_prompt


def construct_verifier_prompt(
    problem_type: str,
    code_snippet: str,
    gold_answer: str,
    predicted_answer: str,
    input_args: str = None,
) -> str:
    """Construct a verification prompt for LLM-based answer comparison."""
    if problem_type == "code_o":
        return VERIFY_OUTPUT_PROMPT.format(
            code_snippet=code_snippet,
            input_args=input_args,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )
    elif problem_type == "code_i":
        return VERIFY_INPUT_PROMPT.format(
            code_snippet=code_snippet,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def construct_verbalized_confidence_prompt(
    problem_type: str,
    code_snippet: str,
    input_args: str,
    gold_output: str,
    answer: str,
) -> str:
    """Construct a prompt to elicit verbalized confidence."""
    if problem_type == "code_o":
        problem_desc = f"Given the code:\n```python\n{code_snippet}\n```\nWith input: {input_args}\nPredict the output."
    else:
        problem_desc = f"Given the code:\n```python\n{code_snippet}\n```\nWith output: {gold_output}\nPredict a valid input."

    return VERBALIZED_CONFIDENCE_PROMPT.format(
        problem_description=problem_desc,
        answer=answer,
    )


def extract_solver_answer(response: str, problem_type: str) -> Optional[str]:
    """Extract the solver's answer from the response."""
    if problem_type == "code_o":
        pattern = r"```output\s*\n?(.*?)\n?```"
    elif problem_type == "code_i":
        pattern = r"```input\s*\n?(.*?)\n?```"
    else:
        return None

    flags = re.DOTALL | re.IGNORECASE
    matches = list(re.finditer(pattern, response, flags))

    if matches:
        return matches[-1].group(1).strip()
    return None


def extract_llm_verdict(response: str) -> Optional[bool]:
    """Extract the LLM's verdict from the verification response."""
    pattern = r"```verdict\s*\n?(CORRECT|INCORRECT)\s*\n?```"
    flags = re.DOTALL | re.IGNORECASE
    matches = list(re.finditer(pattern, response, flags))

    if matches:
        verdict = matches[-1].group(1).strip().upper()
        return verdict == "CORRECT"

    response_upper = response.strip().upper()
    if response_upper.endswith("CORRECT") and not response_upper.endswith("INCORRECT"):
        return True
    elif response_upper.endswith("INCORRECT"):
        return False
    return None


def verify_with_execution(
    executor: PythonExecutor,
    code_snippet: str,
    predicted_answer: str,
    gold_answer: str,
    problem_type: str,
    imports: List[str] = None,
) -> Tuple[bool, str]:
    """Verify the predicted answer by executing Python code."""
    imports = imports or []

    if problem_type == "code_o":
        accuracy = executor.eval_output_prediction(
            code=code_snippet,
            gold_output=gold_answer,
            agent_output=predicted_answer,
            imports=imports,
        )
        return accuracy == 1.0, f"accuracy={accuracy}"

    elif problem_type == "code_i":
        accuracy = executor.eval_input_prediction(
            code=code_snippet,
            gold_output=gold_answer,
            agent_input=predicted_answer,
            imports=imports,
        )
        return accuracy == 1.0, f"accuracy={accuracy}"

    return False, "unknown_problem_type"


def generate_tasks(
    llm: LLM,
    tokenizer: AutoTokenizer,
    seed_data: List[Dict],
    code_f_seed_data: List[Dict],
    problem_types: List[str],
    num_tasks: int,
    config: Dict,
    sampling_params: SamplingParams,
    show_prompts: bool = False,
) -> List[Dict]:
    """Generate proposer tasks."""

    all_tasks = []

    for problem_type in problem_types:
        print(f"\n{'=' * 40}")
        print(f"Problem Type: {problem_type}")
        print(f"{'=' * 40}")

        current_seed_data = code_f_seed_data if problem_type == "code_f" else seed_data

        prompts = []
        references = []
        for i in range(num_tasks):
            prompt, refs = construct_proposer_prompt(
                problem_type=problem_type,
                seed_data=current_seed_data,
                config=config,
            )

            tokens = tokenizer(prompt)["input_ids"]
            if len(tokens) <= config["content_max_length"]:
                prompts.append(prompt)
                references.append(refs)
            else:
                print(f"  [Task {i+1}] Skipped: prompt too long ({len(tokens)} tokens)")

        if not prompts:
            print("  No valid prompts generated!")
            continue

        if show_prompts:
            for i, prompt in enumerate(prompts):
                print(f"\n  --- Proposer Prompt {i+1} ---")
                print("-" * 40)
                print(prompt[:3000] + "..." if len(prompt) > 3000 else prompt)
                print("-" * 40)

        print(f"\n  Generating {len(prompts)} tasks...")
        outputs = llm.generate(prompts, sampling_params)

        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            success, parsed = parse_code_input_output(
                generated_text,
                parse_input=True,
                parse_output=(problem_type == "code_o"),
            )

            task = {
                "task_id": f"{problem_type}_{i}",
                "problem_type": problem_type,
                "proposer_prompt": prompts[i],
                "proposer_response": generated_text,
                "parse_success": success,
                "references": references[i],
            }

            if success:
                task["code_snippet"] = parsed["code"]
                task["input_args"] = parsed["input"]
                task["imports"] = parsed.get("imports", [])

                if problem_type == "code_o":
                    task["gold_output"] = parsed.get("output", "")
                elif problem_type == "code_i":
                    task["gold_input"] = parsed["input"]

            all_tasks.append(task)

            print(f"\n  --- Task {i+1} ---")
            print(f"  Parse success: {success}")
            print(f"  Prompt length: {len(tokenizer(prompts[i])['input_ids'])} tokens")
            print(f"  Generated length: {len(tokenizer(generated_text)['input_ids'])} tokens")

            if success:
                print(f"  Code snippet: {len(parsed['code'])} chars")
                print(f"  Input: {parsed['input'][:100]}...")

            print()
            print("  [GENERATED OUTPUT]:")
            print("-" * 40)
            if len(generated_text) > 2000:
                print(generated_text[:2000])
                print(f"\n  ... [truncated, total {len(generated_text)} chars]")
            else:
                print(generated_text)
            print("-" * 40)

    return all_tasks


def solve_tasks(
    llm: LLM,
    executor: PythonExecutor,
    tasks: List[Dict],
    config: Dict,
    n_samples: int = 1,
    show_prompts: bool = False,
) -> List[Dict]:
    """
    Solve tasks and generate:
    1. Execution label: Ground truth from actually running Python code
    2. LLM label: The LLM judges if solver's answer matches gold answer
    3. Uncertainty metrics: Multiple UQ methods if n_samples > 1
    """

    print("\n" + "=" * 80)
    print("SOLVING TASKS")
    print("=" * 80)

    # Filter to solvable tasks
    solvable_tasks = [t for t in tasks if t.get("parse_success") and t.get("problem_type") in ["code_i", "code_o"]]
    print(f"\n  Solvable tasks: {len(solvable_tasks)} / {len(tasks)}")

    if not solvable_tasks:
        print("  No solvable tasks!")
        return tasks

    # First, compute ground truth for code_i tasks by executing the code
    print("\n  Computing gold outputs for code_i tasks via execution...")
    for task in solvable_tasks:
        if task["problem_type"] == "code_i":
            output, status = executor.run_code(
                code=task["code_snippet"],
                inputs=task["gold_input"],
                imports=task.get("imports", []),
            )
            if "error" not in status.lower():
                task["gold_output"] = output
            else:
                task["gold_output"] = None
                task["execution_error"] = status

    # Create solver prompts
    solver_prompts = []
    solver_task_indices = []

    for idx, task in enumerate(solvable_tasks):
        if task.get("gold_output") is None:
            continue

        if task["problem_type"] == "code_o":
            prompt = construct_solver_prompt(
                problem_type="code_o",
                code_snippet=task["code_snippet"],
                input_args=task["input_args"],
            )
        elif task["problem_type"] == "code_i":
            prompt = construct_solver_prompt(
                problem_type="code_i",
                code_snippet=task["code_snippet"],
                output=task["gold_output"],
            )
        else:
            continue

        solver_prompts.append(prompt)
        solver_task_indices.append(idx)
        task["solver_prompt"] = prompt

    if not solver_prompts:
        print("  No valid solver prompts!")
        return tasks

    if show_prompts:
        print("\n  --- Solver Prompts (first 2) ---")
        for i, prompt in enumerate(solver_prompts[:2]):
            print(f"\n  [Prompt {i+1}]")
            print("-" * 40)
            print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
            print("-" * 40)

    # Generate solver responses with multiple samples for UQ
    print(f"\n  Generating solver responses ({n_samples} samples per task)...")

    solver_sampling_params = SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
        n=n_samples,
        logprobs=1,  # Get log probabilities for UQ
    )

    solver_outputs = llm.generate(solver_prompts, solver_sampling_params)

    # Process solver outputs and compute UQ metrics
    for out_idx, output in enumerate(solver_outputs):
        task_idx = solver_task_indices[out_idx]
        task = solvable_tasks[task_idx]
        problem_type = task["problem_type"]

        # Extract all samples
        all_responses = []
        all_answers = []
        all_logprobs = []

        for sample_output in output.outputs:
            response_text = sample_output.text
            all_responses.append(response_text)

            # Extract answer
            answer = extract_solver_answer(response_text, problem_type)
            all_answers.append(answer)

            # Get log probability info
            cumulative_logprob = sample_output.cumulative_logprob
            n_tokens = len(sample_output.token_ids)
            all_logprobs.append((cumulative_logprob, n_tokens))

        task["solver_responses"] = all_responses
        task["solver_answers"] = all_answers

        # Use first valid answer as the primary answer
        valid_answers = [a for a in all_answers if a is not None]
        task["solver_answer"] = valid_answers[0] if valid_answers else None

        # Compute uncertainty metrics
        if n_samples > 1:
            uq_metrics = compute_all_uncertainty_metrics(all_answers, all_logprobs)
            task["uncertainty_metrics"] = uq_metrics

            # Also store raw data for analysis
            task["logprobs"] = [{"cumulative": lp, "n_tokens": nt} for lp, nt in all_logprobs]

    # Generate LLM verification
    verifier_prompts = []
    verifier_task_indices = []

    for idx, task in enumerate(solvable_tasks):
        if task.get("solver_answer") is None:
            continue

        verifier_prompt = construct_verifier_prompt(
            problem_type=task["problem_type"],
            code_snippet=task["code_snippet"],
            gold_answer=task["gold_output"],
            predicted_answer=task["solver_answer"],
            input_args=task.get("input_args"),
        )

        verifier_prompts.append(verifier_prompt)
        verifier_task_indices.append(idx)
        task["verifier_prompt"] = verifier_prompt

    if show_prompts and verifier_prompts:
        print("\n  --- Verifier Prompts (first 2) ---")
        for i, prompt in enumerate(verifier_prompts[:2]):
            print(f"\n  [Prompt {i+1}]")
            print("-" * 40)
            print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
            print("-" * 40)

    if verifier_prompts:
        print(f"\n  Generating LLM verification for {len(verifier_prompts)} tasks...")

        verifier_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=2048,
            n=1,
        )

        # Time LLM verification
        llm_verify_start = time.time()
        verifier_outputs = llm.generate(verifier_prompts, verifier_sampling_params)
        llm_verify_total_time = time.time() - llm_verify_start

        print(f"  LLM verification time: {llm_verify_total_time:.2f}s total, {llm_verify_total_time/len(verifier_prompts)*1000:.1f}ms per task")

        for out_idx, output in enumerate(verifier_outputs):
            task_idx = verifier_task_indices[out_idx]
            task = solvable_tasks[task_idx]

            verifier_response = output.outputs[0].text
            task["verifier_response"] = verifier_response

            llm_verdict = extract_llm_verdict(verifier_response)
            task["llm_label"] = llm_verdict

            # Store per-task LLM verification time (amortized)
            task["llm_verify_time_ms"] = (llm_verify_total_time / len(verifier_prompts)) * 1000

    # Compute verbalized confidence (optional UQ method)
    if n_samples > 1:
        print("\n  Computing verbalized confidence...")
        confidence_prompts = []
        confidence_task_indices = []

        for idx, task in enumerate(solvable_tasks):
            if task.get("solver_answer") is None:
                continue

            conf_prompt = construct_verbalized_confidence_prompt(
                problem_type=task["problem_type"],
                code_snippet=task["code_snippet"],
                input_args=task.get("input_args", ""),
                gold_output=task.get("gold_output", ""),
                answer=task["solver_answer"],
            )
            confidence_prompts.append(conf_prompt)
            confidence_task_indices.append(idx)

        if confidence_prompts:
            conf_sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=10,
                n=1,
            )

            conf_outputs = llm.generate(confidence_prompts, conf_sampling_params)

            for out_idx, output in enumerate(conf_outputs):
                task_idx = confidence_task_indices[out_idx]
                task = solvable_tasks[task_idx]

                conf_response = output.outputs[0].text
                task["verbalized_confidence_response"] = conf_response

                conf_score = extract_verbalized_confidence(conf_response)
                if "uncertainty_metrics" not in task:
                    task["uncertainty_metrics"] = {}
                task["uncertainty_metrics"]["verbalized_confidence"] = conf_score

    # Compute execution labels (ground truth)
    print("\n  Computing execution labels (ground truth)...")
    exec_verify_total_time = 0.0
    exec_verify_count = 0

    for task in solvable_tasks:
        if task.get("solver_answer") is None:
            task["execution_label"] = None
            task["execution_result"] = "no_solver_answer"
            task["exec_verify_time_ms"] = 0.0
            continue

        # Time execution verification
        exec_start = time.time()
        exec_correct, exec_result = verify_with_execution(
            executor=executor,
            code_snippet=task["code_snippet"],
            predicted_answer=task["solver_answer"],
            gold_answer=task["gold_output"],
            problem_type=task["problem_type"],
            imports=task.get("imports", []),
        )
        exec_time = time.time() - exec_start

        task["execution_label"] = exec_correct
        task["execution_result"] = exec_result
        task["exec_verify_time_ms"] = exec_time * 1000

        exec_verify_total_time += exec_time
        exec_verify_count += 1

        if task.get("llm_label") is not None:
            task["labels_match"] = task["llm_label"] == task["execution_label"]
        else:
            task["labels_match"] = None

    if exec_verify_count > 0:
        print(f"  Execution verification time: {exec_verify_total_time:.2f}s total, {exec_verify_total_time/exec_verify_count*1000:.1f}ms per task")

    # Print results
    print("\n  --- Results ---")
    for task in solvable_tasks:
        if task.get("solver_answer") is not None:
            print(f"\n  Task {task['task_id']}:")
            print(f"    Problem type: {task['problem_type']}")
            print(f"    Solver answer: {str(task['solver_answer'])[:80]}...")
            print(f"    Gold answer: {str(task['gold_output'])[:80]}...")
            print(f"    Execution label (ground truth): {task.get('execution_label')}")
            print(f"    LLM label: {task.get('llm_label')}")
            print(f"    Labels match: {task.get('labels_match')}")
            print(f"    Timing: exec={task.get('exec_verify_time_ms', 0):.1f}ms, llm={task.get('llm_verify_time_ms', 0):.1f}ms")

            if task.get("uncertainty_metrics"):
                print("    Uncertainty metrics:")
                for metric, value in task["uncertainty_metrics"].items():
                    if value is not None and not (isinstance(value, float) and math.isnan(value)):
                        print(f"      - {metric}: {value:.4f}" if isinstance(value, float) else f"      - {metric}: {value}")

    # Print timing summary
    print("\n  --- Timing Summary ---")
    tasks_with_timing = [t for t in solvable_tasks if t.get("exec_verify_time_ms") is not None]
    if tasks_with_timing:
        exec_times = [t.get("exec_verify_time_ms", 0) for t in tasks_with_timing]
        llm_times = [t.get("llm_verify_time_ms", 0) for t in tasks_with_timing]

        print("  Execution verification:")
        print(f"    Total: {sum(exec_times):.1f}ms")
        print(f"    Mean: {np.mean(exec_times):.1f}ms per task")
        print(f"    Median: {np.median(exec_times):.1f}ms per task")
        print(f"    Min/Max: {min(exec_times):.1f}ms / {max(exec_times):.1f}ms")

        print("  LLM verification:")
        print(f"    Total: {sum(llm_times):.1f}ms")
        print(f"    Mean: {np.mean(llm_times):.1f}ms per task")

        if sum(exec_times) > 0:
            speedup = sum(llm_times) / sum(exec_times)
            print(f"\n  LLM/Exec ratio: {speedup:.2f}x")
            if speedup > 1:
                print(f"    (Execution is {1/speedup:.1f}x faster than LLM)")
            else:
                print(f"    (LLM is {1/speedup:.1f}x faster than Execution)")

    return tasks


def _clean_nan_recursive(obj):
    """Recursively replace NaN/Inf with None in nested structures."""
    if isinstance(obj, dict):
        return {k: _clean_nan_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan_recursive(x) for x in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    if hasattr(obj, 'item'):  # numpy scalar
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    return obj


def save_results(tasks: List[Dict], output_path: str):
    """Save results to JSON file."""
    clean_tasks = []
    for task in tasks:
        clean_task = {}
        for k, v in task.items():
            if k == "references":
                clean_task[k] = [dict(ref) for ref in v] if v else []
            else:
                clean_task[k] = _clean_nan_recursive(v)
        clean_tasks.append(clean_task)

    with open(output_path, 'w') as f:
        json.dump(clean_tasks, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate and solve tasks using AZR Coder 3B")
    parser.add_argument("--num_tasks", type=int, default=3, help="Number of tasks to generate per problem type")
    parser.add_argument("--problem_types", nargs="+", default=["code_i", "code_o"],
                        help="Problem types to generate (code_i, code_o, code_f)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for VLLM")

    parser.add_argument("--show_prompts", action="store_true", help="Display prompts")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--solve", action="store_true", help="Enable solver mode with dual labels")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples for UQ (default: 1, use >1 for uncertainty metrics)")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("AZR Task Generation and Solving Analysis")
    print("=" * 80)
    print(f"\nModel: {CONFIG['model_path']}")
    print(f"Problem types: {args.problem_types}")
    print(f"Tasks per type: {args.num_tasks}")
    print(f"Seed: {args.seed}")
    print(f"Solve mode: {args.solve}")
    print(f"N samples for UQ: {args.n_samples}")
    print(f"Show prompts: {args.show_prompts}")
    print(f"Output path: {args.output_path}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"], trust_remote_code=True)

    # Load seed data
    print("Loading seed data...")
    seed_data = load_seed_data(CONFIG["seed_dataset"])
    code_f_seed_data = load_seed_data(CONFIG["code_f_seed_dataset"]) if "code_f" in args.problem_types else []
    print(f"Loaded {len(seed_data)} seed samples for code_i/code_o")
    if code_f_seed_data:
        print(f"Loaded {len(code_f_seed_data)} seed samples for code_f")

    # Initialize VLLM
    print("\nInitializing VLLM...")
    llm = LLM(
        model=CONFIG["model_path"],
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=CONFIG["gpu_memory_utilization"],
        max_num_batched_tokens=CONFIG["max_num_batched_tokens"],
        dtype=CONFIG["dtype"],
        enforce_eager=CONFIG["enforce_eager"],
        trust_remote_code=True,
    )

    proposer_sampling_params = SamplingParams(
        temperature=CONFIG["temperature"],
        top_p=CONFIG["top_p"],
        max_tokens=CONFIG["max_tokens"],
    )

    executor = PythonExecutor(
        timeout_length=CONFIG["execute_max_timeout"],
        ast_check=CONFIG["ast_check"],
    )

    print("\n" + "=" * 80)
    print("GENERATING TASKS")
    print("=" * 80)

    tasks = generate_tasks(
        llm=llm,
        tokenizer=tokenizer,
        seed_data=seed_data,
        code_f_seed_data=code_f_seed_data,
        problem_types=args.problem_types,
        num_tasks=args.num_tasks,
        config=CONFIG,
        sampling_params=proposer_sampling_params,
        show_prompts=args.show_prompts,
    )

    if args.solve:
        tasks = solve_tasks(
            llm=llm,
            executor=executor,
            tasks=tasks,
            config=CONFIG,
            n_samples=args.n_samples,
            show_prompts=args.show_prompts,
        )

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        solvable = [t for t in tasks if t.get("execution_label") is not None]
        if solvable:
            correct_exec = sum(1 for t in solvable if t.get("execution_label", False))
            correct_llm = sum(1 for t in solvable if t.get("llm_label", False))
            labels_match = sum(1 for t in solvable if t.get("labels_match", False))
            llm_label_available = sum(1 for t in solvable if t.get("llm_label") is not None)

            print(f"\n  Total tasks: {len(tasks)}")
            print(f"  Solvable tasks: {len(solvable)}")
            print(f"  Execution correct: {correct_exec}/{len(solvable)} ({100*correct_exec/len(solvable):.1f}%)")
            if llm_label_available > 0:
                print(f"  LLM says correct: {correct_llm}/{llm_label_available} ({100*correct_llm/llm_label_available:.1f}%)")
                print(f"  Labels match: {labels_match}/{llm_label_available} ({100*labels_match/llm_label_available:.1f}%)")

            # Print UQ summary if available
            uq_tasks = [t for t in solvable if t.get("uncertainty_metrics")]
            if uq_tasks:
                print(f"\n  Uncertainty Quantification Summary ({len(uq_tasks)} tasks):")

                # Separate by label match
                match_tasks = [t for t in uq_tasks if t.get("labels_match", False)]
                diff_tasks = [t for t in uq_tasks if t.get("labels_match") is False]

                if match_tasks and diff_tasks:
                    print("\n  Average UQ metrics by label agreement:")
                    all_metrics = set()
                    for t in uq_tasks:
                        all_metrics.update(t["uncertainty_metrics"].keys())

                    for metric in sorted(all_metrics):
                        match_vals = [t["uncertainty_metrics"].get(metric) for t in match_tasks
                                      if t["uncertainty_metrics"].get(metric) is not None
                                      and not (isinstance(t["uncertainty_metrics"].get(metric), float)
                                               and math.isnan(t["uncertainty_metrics"].get(metric)))]
                        diff_vals = [t["uncertainty_metrics"].get(metric) for t in diff_tasks
                                     if t["uncertainty_metrics"].get(metric) is not None
                                     and not (isinstance(t["uncertainty_metrics"].get(metric), float)
                                              and math.isnan(t["uncertainty_metrics"].get(metric)))]

                        if match_vals and diff_vals:
                            match_avg = sum(match_vals) / len(match_vals)
                            diff_avg = sum(diff_vals) / len(diff_vals)
                            print(f"    {metric}:")
                            print(f"      Labels match: {match_avg:.4f}")
                            print(f"      Labels differ: {diff_avg:.4f}")

            # Tasks where labels differ
            diff_tasks = [t for t in solvable if t.get("labels_match") is False]
            if diff_tasks:
                print(f"\n  Tasks where labels differ ({len(diff_tasks)}):")
                for t in diff_tasks:
                    print(f"    - {t['task_id']}: exec={t.get('execution_label')}, llm={t.get('llm_label')}")
                    print(f"      Solver answer: {str(t.get('solver_answer', ''))[:60]}...")
                    print(f"      Gold answer: {str(t.get('gold_output', ''))[:60]}...")

    if args.output_path:
        save_results(tasks, args.output_path)

    executor.cleanup()

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
