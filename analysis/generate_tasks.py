"""
Task Generation and Solving Analysis Script for Absolute Zero Reasoner

This script generates tasks using the challenger/proposer from a pretrained model
and evaluates solver performance using both code execution and LLM-based verification.

Two labels are produced for each task:
1. Execution label: Ground truth from actually running the Python code
2. LLM label: The LLM judges if the solver's answer matches the gold answer (no code execution)

Usage:
    # Generate tasks only
    python -m analysis.generate_tasks --num_tasks 10

    # Generate and solve tasks with both labels
    python -m analysis.generate_tasks --num_tasks 5 --solve

    # Show prompts and save results
    python -m analysis.generate_tasks --num_tasks 5 --solve --show_prompts --output_path results.json
"""

import os
import sys
import json
import argparse
import re
from typing import List, Dict, Tuple, Optional
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
- Different string representations of the same value should be considered equivalent (e.g., "1" and 1, or {'a': 1} and {{'a': 1}})
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
        # For code_f, we use a single reference (the code snippet to generate inputs for)
        # code_f seeds have 'snippet', 'inputs', 'outputs', 'message' format
        chosen_ref = random.choice(seed_data, size=1, replace=False)[0]
        # Convert to the format expected by the prompt generator
        reference_snippets = [{
            'snippet': chosen_ref['snippet'],
            'input': chosen_ref.get('inputs', [''])[0] if chosen_ref.get('inputs') else '',
            'output': chosen_ref.get('outputs', [''])[0] if chosen_ref.get('outputs') else '',
            'imports': chosen_ref.get('imports', []),
        }]
    else:
        # For code_i and code_o, sample multiple references
        chosen_references = random.choice(
            seed_data,
            size=min(config["io_n"], len(seed_data)),
            replace=False
        ).tolist()
        reference_snippets = chosen_references

    # Get the generator prompt
    generator_prompt = get_code_problem_generator_prompt(
        problem_type=problem_type,
        reference_snippets=reference_snippets,
        banned_keywords=config["banned_keywords"],
        banned_assertion_keywords=config["banned_assertion_keywords"],
        composite_functions=[],  # No composite functions for analysis
        remove_after_return=False,
        num_inputs=config["num_inputs"],
        remove_input_from_snippet=False,
    )

    # Wrap with instruction template (matching reward_fn.extraction_type=answer_conditional)
    full_prompt = instruction_following.format(generator_prompt)

    return full_prompt, reference_snippets


def construct_solver_prompt(
    problem_type: str,
    code_snippet: str,
    input_args: str = None,
    output: str = None,
) -> str:
    """
    Construct a solver prompt for a given task.

    For code_o (output prediction): Given code and input, predict output
    For code_i (input prediction): Given code and output, predict input
    """
    solver_prompt = get_code_problem_predictor_prompt(
        problem_type=problem_type,
        snippet=code_snippet,
        input_args=input_args,
        output=output,
    )

    # Wrap with instruction template
    full_prompt = instruction_following.format(solver_prompt)

    return full_prompt


def construct_verifier_prompt(
    problem_type: str,
    code_snippet: str,
    gold_answer: str,
    predicted_answer: str,
    input_args: str = None,
) -> str:
    """
    Construct a verification prompt for LLM-based answer comparison.

    The LLM will judge if the predicted answer is correct without executing code.
    """
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


def extract_solver_answer(response: str, problem_type: str) -> Optional[str]:
    """
    Extract the solver's answer from the response.

    For code_o: Extract from ```output``` block
    For code_i: Extract from ```input``` block
    """
    if problem_type == "code_o":
        pattern = r"```output\s*\n?(.*?)\n?```"
    elif problem_type == "code_i":
        pattern = r"```input\s*\n?(.*?)\n?```"
    else:
        return None

    flags = re.DOTALL | re.IGNORECASE
    matches = list(re.finditer(pattern, response, flags))

    if matches:
        # Take the last match (in case there are multiple)
        return matches[-1].group(1).strip()

    return None


def extract_llm_verdict(response: str) -> Optional[bool]:
    """
    Extract the LLM's verdict from the verification response.

    Returns: True if CORRECT, False if INCORRECT, None if cannot parse
    """
    pattern = r"```verdict\s*\n?(CORRECT|INCORRECT)\s*\n?```"
    flags = re.DOTALL | re.IGNORECASE
    matches = list(re.finditer(pattern, response, flags))

    if matches:
        verdict = matches[-1].group(1).strip().upper()
        return verdict == "CORRECT"

    # Fallback: check if response ends with CORRECT or INCORRECT
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
    """
    Verify the predicted answer by executing Python code.

    For code_o: predicted_answer is the solver's predicted output, gold_answer is the true output
    For code_i: predicted_answer is the solver's predicted input, gold_answer is the true output

    Returns: (is_correct, execution_result)
    """
    imports = imports or []

    if problem_type == "code_o":
        # For output prediction: check if predicted output matches gold output
        accuracy = executor.eval_output_prediction(
            code=code_snippet,
            gold_output=gold_answer,
            agent_output=predicted_answer,
            imports=imports,
        )
        return accuracy == 1.0, f"accuracy={accuracy}"

    elif problem_type == "code_i":
        # For input prediction: check if predicted input produces the gold output
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

        # Select appropriate seed data
        current_seed_data = code_f_seed_data if problem_type == "code_f" else seed_data

        # Generate prompts
        prompts = []
        references = []
        for i in range(num_tasks):
            prompt, refs = construct_proposer_prompt(
                problem_type=problem_type,
                seed_data=current_seed_data,
                config=config,
            )

            # Check token length
            tokens = tokenizer(prompt)["input_ids"]
            if len(tokens) <= config["content_max_length"]:
                prompts.append(prompt)
                references.append(refs)
            else:
                print(f"  [Task {i+1}] Skipped: prompt too long ({len(tokens)} tokens)")

        if not prompts:
            print("  No valid prompts generated!")
            continue

        # Show prompts if requested
        if show_prompts:
            for i, prompt in enumerate(prompts):
                print(f"\n  --- Proposer Prompt {i+1} ---")
                print("-" * 40)
                print(prompt[:3000] + "..." if len(prompt) > 3000 else prompt)
                print("-" * 40)

        # Generate with VLLM
        print(f"\n  Generating {len(prompts)} tasks...")
        outputs = llm.generate(prompts, sampling_params)

        # Process results
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            # Parse the generated output
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
                    # For code_o, the proposer provides the output
                    task["gold_output"] = parsed.get("output", "")
                elif problem_type == "code_i":
                    # For code_i, we need to execute to get the output
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
    show_prompts: bool = False,
) -> List[Dict]:
    """
    Solve tasks and generate two labels:
    1. Execution label: Ground truth from actually running Python code
    2. LLM label: The LLM judges if solver's answer matches gold answer (no code execution)
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
            # Execute to get the gold output
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
            # Solver predicts output from code + input
            prompt = construct_solver_prompt(
                problem_type="code_o",
                code_snippet=task["code_snippet"],
                input_args=task["input_args"],
            )
        elif task["problem_type"] == "code_i":
            # Solver predicts input from code + output
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

    # Show prompts if requested
    if show_prompts:
        print("\n  --- Solver Prompts (first 2) ---")
        for i, prompt in enumerate(solver_prompts[:2]):
            print(f"\n  [Prompt {i+1}]")
            print("-" * 40)
            print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
            print("-" * 40)

    # Generate solver responses (single sample per task)
    print(f"\n  Generating solver responses for {len(solver_prompts)} tasks...")

    solver_sampling_params = SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
        n=1,  # Single sample
    )

    solver_outputs = llm.generate(solver_prompts, solver_sampling_params)

    # Extract solver answers
    for out_idx, output in enumerate(solver_outputs):
        task_idx = solver_task_indices[out_idx]
        task = solvable_tasks[task_idx]
        problem_type = task["problem_type"]

        response_text = output.outputs[0].text
        task["solver_response"] = response_text

        # Extract the answer
        solver_answer = extract_solver_answer(response_text, problem_type)
        task["solver_answer"] = solver_answer

    # Now create verification prompts for LLM-based label
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

    # Show verifier prompts if requested
    if show_prompts and verifier_prompts:
        print("\n  --- Verifier Prompts (first 2) ---")
        for i, prompt in enumerate(verifier_prompts[:2]):
            print(f"\n  [Prompt {i+1}]")
            print("-" * 40)
            print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
            print("-" * 40)

    # Generate LLM verification responses
    if verifier_prompts:
        print(f"\n  Generating LLM verification for {len(verifier_prompts)} tasks...")

        verifier_sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic for verification
            top_p=1.0,
            max_tokens=2048,  # Shorter for verification
            n=1,
        )

        verifier_outputs = llm.generate(verifier_prompts, verifier_sampling_params)

        # Extract LLM verdicts
        for out_idx, output in enumerate(verifier_outputs):
            task_idx = verifier_task_indices[out_idx]
            task = solvable_tasks[task_idx]

            verifier_response = output.outputs[0].text
            task["verifier_response"] = verifier_response

            # Extract the verdict
            llm_verdict = extract_llm_verdict(verifier_response)
            task["llm_label"] = llm_verdict

    # Compute execution labels (ground truth)
    print("\n  Computing execution labels (ground truth)...")
    for task in solvable_tasks:
        if task.get("solver_answer") is None:
            task["execution_label"] = None
            task["execution_result"] = "no_solver_answer"
            continue

        exec_correct, exec_result = verify_with_execution(
            executor=executor,
            code_snippet=task["code_snippet"],
            predicted_answer=task["solver_answer"],
            gold_answer=task["gold_output"],
            problem_type=task["problem_type"],
            imports=task.get("imports", []),
        )

        task["execution_label"] = exec_correct
        task["execution_result"] = exec_result

        # Check if labels match
        if task.get("llm_label") is not None:
            task["labels_match"] = task["llm_label"] == task["execution_label"]
        else:
            task["labels_match"] = None

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

    return tasks


def save_results(tasks: List[Dict], output_path: str):
    """Save results to JSON file."""
    # Clean up non-serializable objects
    clean_tasks = []
    for task in tasks:
        clean_task = {}
        for k, v in task.items():
            if k == "references":
                # Convert numpy arrays if any
                clean_task[k] = [dict(ref) for ref in v] if v else []
            else:
                clean_task[k] = v
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

    # New arguments
    parser.add_argument("--show_prompts", action="store_true", help="Display prompts")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--solve", action="store_true", help="Enable solver mode with dual labels")

    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 80)
    print("AZR Task Generation and Solving Analysis")
    print("=" * 80)
    print(f"\nModel: {CONFIG['model_path']}")
    print(f"Problem types: {args.problem_types}")
    print(f"Tasks per type: {args.num_tasks}")
    print(f"Seed: {args.seed}")
    print(f"Solve mode: {args.solve}")
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

    # Sampling parameters for proposer
    proposer_sampling_params = SamplingParams(
        temperature=CONFIG["temperature"],
        top_p=CONFIG["top_p"],
        max_tokens=CONFIG["max_tokens"],
    )

    # Initialize executor for code verification
    executor = PythonExecutor(
        timeout_length=CONFIG["execute_max_timeout"],
        ast_check=CONFIG["ast_check"],
    )

    print("\n" + "=" * 80)
    print("GENERATING TASKS")
    print("=" * 80)

    # Generate tasks
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

    # Solve tasks if requested
    if args.solve:
        tasks = solve_tasks(
            llm=llm,
            executor=executor,
            tasks=tasks,
            config=CONFIG,
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
            print(f"  LLM says correct: {correct_llm}/{llm_label_available} ({100*correct_llm/llm_label_available:.1f}%)" if llm_label_available > 0 else "  LLM labels: N/A")
            print(f"  Labels match: {labels_match}/{llm_label_available} ({100*labels_match/llm_label_available:.1f}%)" if llm_label_available > 0 else "  Labels match: N/A")

            # Tasks where labels differ
            diff_tasks = [t for t in solvable if t.get("labels_match") is False]
            if diff_tasks:
                print(f"\n  Tasks where labels differ ({len(diff_tasks)}):")
                for t in diff_tasks:
                    print(f"    - {t['task_id']}: exec={t.get('execution_label')}, llm={t.get('llm_label')}")
                    print(f"      Solver answer: {str(t.get('solver_answer', ''))[:60]}...")
                    print(f"      Gold answer: {str(t.get('gold_output', ''))[:60]}...")

    # Save results if requested
    if args.output_path:
        save_results(tasks, args.output_path)

    # Cleanup
    executor.cleanup()

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
