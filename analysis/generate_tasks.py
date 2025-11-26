"""
Task Generation Analysis Script for Absolute Zero Reasoner

This script generates tasks using the challenger/proposer from a pretrained model
using VLLM for inference. Uses exact hyperparameters from the 3B coder model.

Usage:
    python -m analysis.generate_tasks --num_tasks 10
"""

import os
import sys
import json
import argparse
from typing import List, Dict
from numpy import random

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from absolute_zero_reasoner.data_construction.prompts import get_code_problem_generator_prompt
from absolute_zero_reasoner.data_construction.process_data import instruction_following


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
    tokenizer: AutoTokenizer,
) -> str:
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


def main():
    parser = argparse.ArgumentParser(description="Generate tasks using AZR Coder 3B")
    parser.add_argument("--num_tasks", type=int, default=3, help="Number of tasks to generate per problem type")
    parser.add_argument("--problem_types", nargs="+", default=["code_i", "code_o"],
                        help="Problem types to generate (code_i, code_o, code_f)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for VLLM")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 80)
    print("AZR Task Generation Analysis")
    print("=" * 80)
    print(f"\nModel: {CONFIG['model_path']}")
    print(f"Problem types: {args.problem_types}")
    print(f"Tasks per type: {args.num_tasks}")
    print(f"Seed: {args.seed}")
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

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=CONFIG["temperature"],
        top_p=CONFIG["top_p"],
        max_tokens=CONFIG["max_tokens"],
    )

    print("\n" + "=" * 80)
    print("GENERATING TASKS")
    print("=" * 80)

    for problem_type in args.problem_types:
        print(f"\n{'=' * 40}")
        print(f"Problem Type: {problem_type}")
        print(f"{'=' * 40}")

        # Select appropriate seed data
        current_seed_data = code_f_seed_data if problem_type == "code_f" else seed_data

        # Generate prompts
        prompts = []
        references = []
        for i in range(args.num_tasks):
            prompt, refs = construct_proposer_prompt(
                problem_type=problem_type,
                seed_data=current_seed_data,
                config=CONFIG,
                tokenizer=tokenizer,
            )

            # Check token length
            tokens = tokenizer(prompt)["input_ids"]
            if len(tokens) <= CONFIG["content_max_length"]:
                prompts.append(prompt)
                references.append(refs)
            else:
                print(f"  [Task {i+1}] Skipped: prompt too long ({len(tokens)} tokens)")

        if not prompts:
            print("  No valid prompts generated!")
            continue

        # Generate with VLLM
        print(f"\n  Generating {len(prompts)} tasks...")
        outputs = llm.generate(prompts, sampling_params)

        # Print results
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            print(f"\n  --- Task {i+1} ---")
            print(f"  Prompt length: {len(tokenizer(prompts[i])['input_ids'])} tokens")
            print(f"  Generated length: {len(tokenizer(generated_text)['input_ids'])} tokens")
            print()
            print("  [GENERATED OUTPUT]:")
            print("-" * 40)
            # Print first 2000 chars for readability
            if len(generated_text) > 2000:
                print(generated_text[:2000])
                print(f"\n  ... [truncated, total {len(generated_text)} chars]")
            else:
                print(generated_text)
            print("-" * 40)

    print("\n" + "=" * 80)
    print("TASK GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
