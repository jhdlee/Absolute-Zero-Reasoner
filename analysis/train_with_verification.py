"""
Training Script with Different Verification Modes

This script runs initial training steps using three different verification strategies:
1. FULL_EXECUTION: Always use Python executor (ground truth)
2. FULL_LLM: Always use LLM-as-a-judge
3. ADAPTIVE: Use uncertainty to decide, with a budget constraint

Usage:
    # Run with full execution (baseline)
    python -m analysis.train_with_verification --mode full_execution --num_steps 100

    # Run with full LLM
    python -m analysis.train_with_verification --mode full_llm --num_steps 100

    # Run with adaptive (our method)
    python -m analysis.train_with_verification --mode adaptive --budget 0.3 --num_steps 100

    # Run all three modes for comparison
    python -m analysis.train_with_verification --mode all --num_steps 100
"""

import os
import sys
import json
import argparse
import time
import math
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np

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

from analysis.adaptive_reward_manager import (
    AdaptiveRewardManager,
    VerificationMode,
    create_reward_manager,
)


# ==============================================================================
# Configuration
# ==============================================================================

CONFIG = {
    "model_name": "andrewzh/Absolute_Zero_Reasoner-Coder-3b",
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 8096,
    "io_n": 6,  # Number of reference snippets
    "content_max_length": 8096,
    "seed_data_path": "data/3b_coder_seed_io.jsonl",
    # Banned keywords (from coder config)
    "banned_keywords": [
        "logging", "random", "multiprocessing", "pebble", "subprocess",
        "threading", "datetime", "time", "hashlib", "hmac", "bcrypt",
        "os.sys", "os.path", "sys.exit", "os.environ", "calendar"
    ],
    "banned_assertion_keywords": [],
}


# ==============================================================================
# Data Loading
# ==============================================================================

def load_seed_data(path: str) -> List[Dict]:
    """Load seed data from JSONL file."""
    data = []
    full_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        path
    )

    if not os.path.exists(full_path):
        print(f"Warning: Seed data not found at {full_path}")
        return []

    with open(full_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return data


# ==============================================================================
# Task Generation
# ==============================================================================

def generate_proposer_tasks(
    llm: LLM,
    tokenizer: AutoTokenizer,
    seed_data: List[Dict],
    problem_types: List[str],
    num_tasks: int,
    config: Dict,
) -> List[Dict]:
    """Generate tasks using the proposer."""
    from numpy import random

    all_tasks = []

    for problem_type in problem_types:
        print(f"\n  Generating {num_tasks} {problem_type} tasks...")

        prompts = []
        for i in range(num_tasks):
            # Sample reference snippets
            reference_snippets = random.choice(
                seed_data,
                size=min(config["io_n"], len(seed_data)),
                replace=False
            ).tolist()

            # Construct prompt using the correct API
            if problem_type in ["code_i", "code_o"]:
                generator_prompt = get_code_problem_generator_prompt(
                    problem_type=problem_type,
                    reference_snippets=reference_snippets,
                    banned_keywords=config["banned_keywords"],
                    banned_assertion_keywords=config["banned_assertion_keywords"],
                    composite_functions=[],
                    remove_after_return=False,
                    num_inputs=10,
                    remove_input_from_snippet=False,
                )
                prompt = instruction_following.format(generator_prompt)
            else:
                continue

            tokens = tokenizer(prompt)["input_ids"]
            if len(tokens) <= config["content_max_length"]:
                prompts.append(prompt)

        if not prompts:
            continue

        # Generate
        sampling_params = SamplingParams(
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=config["max_tokens"],
            n=1,
        )

        outputs = llm.generate(prompts, sampling_params)

        # Parse outputs
        for i, output in enumerate(outputs):
            response = output.outputs[0].text

            success, parsed = parse_code_input_output(
                response,
                parse_input=True,
                parse_output=(problem_type == "code_o"),
            )

            if success:
                task = {
                    "task_id": f"{problem_type}_{len(all_tasks)}",
                    "problem_type": problem_type,
                    "code_snippet": parsed["code"],
                    "input_args": parsed["input"],
                    "imports": parsed.get("imports", []),
                    "parse_success": True,
                }

                if problem_type == "code_o":
                    task["gold_output"] = parsed.get("output", "")
                elif problem_type == "code_i":
                    task["gold_input"] = parsed["input"]

                all_tasks.append(task)

    return all_tasks


def compute_gold_outputs(
    executor: PythonExecutor,
    tasks: List[Dict],
) -> List[Dict]:
    """Compute gold outputs for code_i tasks by executing code."""
    for task in tasks:
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

    return tasks


# ==============================================================================
# Solver Generation
# ==============================================================================

def generate_solver_responses(
    llm: LLM,
    tasks: List[Dict],
    n_samples: int = 1,
) -> List[Dict]:
    """Generate solver responses for tasks."""
    prompts = []
    task_indices = []

    for idx, task in enumerate(tasks):
        if task.get("gold_output") is None:
            continue

        problem_type = task["problem_type"]
        code = task["code_snippet"]

        if problem_type == "code_o":
            input_args = task["input_args"]
            prompt = f"""Given the following Python code and input, predict the output.

```python
{code}
```

Input: {input_args}

What is the output? Provide only the output value, nothing else."""

        elif problem_type == "code_i":
            gold_output = task["gold_output"]
            prompt = f"""Given the following Python code and expected output, predict the input.

```python
{code}
```

Expected Output: {gold_output}

What input would produce this output? Provide only the input value, nothing else."""
        else:
            continue

        prompts.append(prompt)
        task_indices.append(idx)
        task["solver_prompt"] = prompt

    if not prompts:
        return tasks

    # Generate
    sampling_params = SamplingParams(
        temperature=0.7 if n_samples > 1 else 0.0,
        top_p=0.95,
        max_tokens=1024,
        n=n_samples,
    )

    outputs = llm.generate(prompts, sampling_params)

    # Parse outputs
    for out_idx, output in enumerate(outputs):
        task_idx = task_indices[out_idx]
        task = tasks[task_idx]

        responses = [o.text.strip() for o in output.outputs]
        task["solver_responses"] = responses

        # Extract primary answer
        primary_answer = extract_answer(responses[0])
        task["solver_answer"] = primary_answer

        # If multiple samples, store all answers for UQ
        if n_samples > 1:
            all_answers = [extract_answer(r) for r in responses]
            task["solver_answers"] = all_answers

    return tasks


def extract_answer(response: str) -> Optional[str]:
    """Extract answer from solver response."""
    response = response.strip()

    # Remove common prefixes
    prefixes = ["Output:", "Answer:", "Result:", "Input:"]
    for prefix in prefixes:
        if response.lower().startswith(prefix.lower()):
            response = response[len(prefix):].strip()

    # Take first line
    lines = response.split("\n")
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            return line

    return response if response else None


# ==============================================================================
# Training Step Simulation
# ==============================================================================

def run_training_step(
    reward_manager: AdaptiveRewardManager,
    tasks: List[Dict],
    problem_type: str,
) -> Dict:
    """
    Run a single training step and collect metrics.

    Returns:
        Dict with step results and metrics
    """
    # Filter valid tasks
    valid_tasks = [
        t for t in tasks
        if t.get("solver_answer") is not None and t.get("gold_output") is not None
    ]

    if not valid_tasks:
        return {"error": "no_valid_tasks", "n_tasks": 0}

    # Compute rewards
    start_time = time.time()
    results = reward_manager.compute_rewards(valid_tasks, problem_type)
    total_time = time.time() - start_time

    # Collect metrics
    rewards = [r["reward"] for r in results]
    correct = [r["verification_correct"] for r in results if r["verification_correct"] is not None]

    step_results = {
        "n_tasks": len(valid_tasks),
        "n_verified": len(correct),
        "mean_reward": np.mean(rewards) if rewards else 0,
        "std_reward": np.std(rewards) if rewards else 0,
        "accuracy": np.mean(correct) if correct else 0,
        "total_time_s": total_time,
        "time_per_task_ms": (total_time / len(valid_tasks)) * 1000 if valid_tasks else 0,
        "verification_stats": reward_manager.get_stats_summary(),
        "results": results,
    }

    return step_results


# ==============================================================================
# Ground Truth Comparison
# ==============================================================================

def compute_ground_truth_labels(
    executor: PythonExecutor,
    tasks: List[Dict],
) -> List[Dict]:
    """Compute ground truth labels using execution for all tasks."""
    for task in tasks:
        if task.get("solver_answer") is None or task.get("gold_output") is None:
            task["ground_truth_correct"] = None
            continue

        problem_type = task["problem_type"]
        code = task["code_snippet"]
        predicted = task["solver_answer"]
        gold = task["gold_output"]
        imports = task.get("imports", [])

        try:
            if problem_type == "code_o":
                accuracy = executor.eval_output_prediction(
                    code=code,
                    gold_output=gold,
                    agent_output=str(predicted),
                    imports=imports,
                )
            elif problem_type == "code_i":
                accuracy = executor.eval_input_prediction(
                    code=code,
                    gold_output=gold,
                    agent_input=str(predicted),
                    imports=imports,
                )
            else:
                accuracy = 0

            task["ground_truth_correct"] = (accuracy == 1.0)

        except Exception as e:
            task["ground_truth_correct"] = None
            task["ground_truth_error"] = str(e)

    return tasks


def compare_with_ground_truth(results: List[Dict]) -> Dict:
    """Compare verification results with ground truth."""
    comparisons = {
        "total": 0,
        "agreements": 0,
        "disagreements": 0,
        "false_positives": 0,  # Verified correct but actually wrong
        "false_negatives": 0,  # Verified wrong but actually correct
        "execution_correct": 0,
        "llm_correct": 0,
    }

    for r in results:
        gt = r.get("ground_truth_correct")
        verified = r.get("verification_correct")
        method = r.get("verification_method")

        if gt is None or verified is None:
            continue

        comparisons["total"] += 1

        if gt == verified:
            comparisons["agreements"] += 1
            if method == "execution":
                comparisons["execution_correct"] += 1
            elif method == "llm":
                comparisons["llm_correct"] += 1
        else:
            comparisons["disagreements"] += 1
            if verified and not gt:
                comparisons["false_positives"] += 1
            elif not verified and gt:
                comparisons["false_negatives"] += 1

    if comparisons["total"] > 0:
        comparisons["agreement_rate"] = comparisons["agreements"] / comparisons["total"]
        comparisons["false_positive_rate"] = comparisons["false_positives"] / comparisons["total"]
        comparisons["false_negative_rate"] = comparisons["false_negatives"] / comparisons["total"]

    return comparisons


# ==============================================================================
# Main Training Loop
# ==============================================================================

def run_experiment(
    mode: str,
    num_steps: int,
    tasks_per_step: int,
    problem_types: List[str],
    budget_fraction: float = 0.3,
    n_samples_for_uq: int = 8,
    output_dir: str = "training_results",
    tensor_parallel_size: int = 1,
) -> Dict:
    """
    Run training experiment with specified verification mode.

    Args:
        mode: "full_execution", "full_llm", or "adaptive"
        num_steps: Number of training steps to run
        tasks_per_step: Number of tasks per step
        problem_types: List of problem types
        budget_fraction: For adaptive mode
        n_samples_for_uq: For adaptive mode
        output_dir: Where to save results
        tensor_parallel_size: VLLM tensor parallel size

    Returns:
        Experiment results dict
    """
    print("=" * 80)
    print(f"TRAINING EXPERIMENT: {mode.upper()}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    print("\nInitializing model and executor...")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    llm = LLM(
        model=CONFIG["model_name"],
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )
    executor = PythonExecutor(timeout_length=10)

    # Load seed data
    seed_data = load_seed_data(CONFIG["seed_data_path"])
    print(f"  Loaded {len(seed_data)} seed examples")

    # Create reward manager
    reward_manager = create_reward_manager(
        mode=mode,
        executor=executor,
        llm=llm,
        budget_fraction=budget_fraction,
        n_samples_for_uq=n_samples_for_uq,
    )

    # Run training steps
    all_results = []
    step_metrics = []

    print(f"\nRunning {num_steps} training steps...")

    for step in range(num_steps):
        print(f"\n--- Step {step + 1}/{num_steps} ---")

        # Reset reward manager stats for this step
        reward_manager.reset_stats()

        # Generate tasks
        print("  Generating tasks...")
        tasks = generate_proposer_tasks(
            llm=llm,
            tokenizer=tokenizer,
            seed_data=seed_data,
            problem_types=problem_types,
            num_tasks=tasks_per_step,
            config=CONFIG,
        )
        print(f"    Generated {len(tasks)} tasks")

        if not tasks:
            print("    No tasks generated, skipping step")
            continue

        # Compute gold outputs for code_i tasks
        tasks = compute_gold_outputs(executor, tasks)

        # Generate solver responses
        print("  Generating solver responses...")
        n_samples = n_samples_for_uq if mode == "adaptive" else 1
        tasks = generate_solver_responses(llm, tasks, n_samples=n_samples)

        # Run training step for each problem type
        step_results = {}
        for problem_type in problem_types:
            type_tasks = [t for t in tasks if t["problem_type"] == problem_type]
            if not type_tasks:
                continue

            print(f"  Verifying {len(type_tasks)} {problem_type} tasks...")
            result = run_training_step(reward_manager, type_tasks, problem_type)
            step_results[problem_type] = result

            # Compute ground truth for comparison
            if mode != "full_execution":
                print("  Computing ground truth for comparison...")
                type_tasks = compute_ground_truth_labels(executor, type_tasks)
                comparison = compare_with_ground_truth(result.get("results", []))
                step_results[problem_type]["ground_truth_comparison"] = comparison

        # Aggregate metrics
        total_tasks = sum(r.get("n_tasks", 0) for r in step_results.values())
        total_time = sum(r.get("total_time_s", 0) for r in step_results.values())
        mean_reward = np.mean([
            r.get("mean_reward", 0) for r in step_results.values() if r.get("n_tasks", 0) > 0
        ])

        step_metric = {
            "step": step + 1,
            "n_tasks": total_tasks,
            "total_time_s": total_time,
            "mean_reward": mean_reward,
            "verification_stats": reward_manager.get_stats_summary(),
            "per_type": step_results,
        }
        step_metrics.append(step_metric)

        print(f"  Mean reward: {mean_reward:.4f}")
        print(f"  Total time: {total_time:.2f}s")

        # Store detailed results
        all_results.extend([
            {**r, "step": step + 1}
            for type_result in step_results.values()
            for r in type_result.get("results", [])
        ])

    # Compute aggregate statistics
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    aggregate_stats = {
        "mode": mode,
        "num_steps": num_steps,
        "tasks_per_step": tasks_per_step,
        "problem_types": problem_types,
        "total_tasks": sum(m.get("n_tasks", 0) for m in step_metrics),
        "total_time_s": sum(m.get("total_time_s", 0) for m in step_metrics),
        "mean_reward": np.mean([m.get("mean_reward", 0) for m in step_metrics]),
        "std_reward": np.std([m.get("mean_reward", 0) for m in step_metrics]),
    }

    # Verification breakdown
    final_stats = reward_manager.get_stats_summary()
    aggregate_stats["verification_breakdown"] = {
        "execution_verifications": final_stats.get("execution_verifications", 0),
        "llm_verifications": final_stats.get("llm_verifications", 0),
        "execution_time_ms": final_stats.get("execution_time_ms", 0),
        "llm_time_ms": final_stats.get("llm_time_ms", 0),
    }

    # Ground truth comparison (for LLM and adaptive modes)
    if mode != "full_execution":
        gt_comparisons = []
        for m in step_metrics:
            for type_result in m.get("per_type", {}).values():
                gt = type_result.get("ground_truth_comparison")
                if gt:
                    gt_comparisons.append(gt)

        if gt_comparisons:
            aggregate_stats["ground_truth_accuracy"] = np.mean([
                c.get("agreement_rate", 0) for c in gt_comparisons
            ])
            aggregate_stats["false_positive_rate"] = np.mean([
                c.get("false_positive_rate", 0) for c in gt_comparisons
            ])
            aggregate_stats["false_negative_rate"] = np.mean([
                c.get("false_negative_rate", 0) for c in gt_comparisons
            ])

    print(f"\nMode: {mode}")
    print(f"Total tasks: {aggregate_stats['total_tasks']}")
    print(f"Total time: {aggregate_stats['total_time_s']:.2f}s")
    print(f"Mean reward: {aggregate_stats['mean_reward']:.4f} ± {aggregate_stats['std_reward']:.4f}")

    if "ground_truth_accuracy" in aggregate_stats:
        print(f"Ground truth accuracy: {aggregate_stats['ground_truth_accuracy']:.2%}")
        print(f"False positive rate: {aggregate_stats['false_positive_rate']:.2%}")
        print(f"False negative rate: {aggregate_stats['false_negative_rate']:.2%}")

    # Save results
    experiment_results = {
        "config": {
            "mode": mode,
            "num_steps": num_steps,
            "tasks_per_step": tasks_per_step,
            "problem_types": problem_types,
            "budget_fraction": budget_fraction,
            "n_samples_for_uq": n_samples_for_uq,
        },
        "aggregate_stats": aggregate_stats,
        "step_metrics": step_metrics,
    }

    output_path = os.path.join(output_dir, f"experiment_{mode}.json")
    with open(output_path, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Save detailed results
    detailed_path = os.path.join(output_dir, f"detailed_{mode}.json")
    with open(detailed_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Detailed results saved to: {detailed_path}")

    return experiment_results


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run training with different verification modes"
    )
    parser.add_argument(
        "--mode", type=str, default="full_execution",
        choices=["full_execution", "full_llm", "adaptive", "all"],
        help="Verification mode (or 'all' to run all three)"
    )
    parser.add_argument(
        "--num_steps", type=int, default=10,
        help="Number of training steps"
    )
    parser.add_argument(
        "--tasks_per_step", type=int, default=20,
        help="Number of tasks per step"
    )
    parser.add_argument(
        "--problem_types", nargs="+", default=["code_i", "code_o"],
        help="Problem types to train on"
    )
    parser.add_argument(
        "--budget", type=float, default=0.3,
        help="Budget fraction for adaptive mode"
    )
    parser.add_argument(
        "--n_samples_uq", type=int, default=8,
        help="Number of samples for UQ in adaptive mode"
    )
    parser.add_argument(
        "--output_dir", type=str, default="training_results",
        help="Output directory"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="Tensor parallel size for VLLM"
    )

    args = parser.parse_args()

    if args.mode == "all":
        # Run all three modes
        modes = ["full_execution", "full_llm", "adaptive"]
        all_results = {}

        for mode in modes:
            result = run_experiment(
                mode=mode,
                num_steps=args.num_steps,
                tasks_per_step=args.tasks_per_step,
                problem_types=args.problem_types,
                budget_fraction=args.budget,
                n_samples_for_uq=args.n_samples_uq,
                output_dir=args.output_dir,
                tensor_parallel_size=args.tensor_parallel_size,
            )
            all_results[mode] = result

        # Save comparison
        comparison_path = os.path.join(args.output_dir, "comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nComparison saved to: {comparison_path}")

    else:
        # Run single mode
        run_experiment(
            mode=args.mode,
            num_steps=args.num_steps,
            tasks_per_step=args.tasks_per_step,
            problem_types=args.problem_types,
            budget_fraction=args.budget,
            n_samples_for_uq=args.n_samples_uq,
            output_dir=args.output_dir,
            tensor_parallel_size=args.tensor_parallel_size,
        )


if __name__ == "__main__":
    main()
