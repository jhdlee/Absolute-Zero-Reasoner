#!/bin/bash
# Run uncertainty analysis with 3B coder hyperparameters from the paper
# Uses vLLM for fast inference
#
# Usage:
#   ./analysis/run_uncertainty_analysis.sh [OPTIONS]
#
# Options:
#   --quick      Run with fewer tasks (20 instead of 100) for quick testing
#   --skip-gen   Skip task generation, use seed data directly as tasks
#   --tp N       Set tensor parallel size (default: 2)

set -x

# Default parameters matching coder3b.sh (using trained checkpoint)
MODEL_PATH="andrewzh/Absolute_Zero_Reasoner-Coder-3b"
SEED_PATH="data/3b_coder_seed_io.jsonl"
NUM_TASKS=100
NUM_SOLVER_SAMPLES=8
OUTPUT_DIR="analysis/results"
PROBLEM_TYPES="code_i code_o"  # code_f excluded for simpler analysis by default
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.4

# Parse arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            NUM_TASKS=20
            NUM_SOLVER_SAMPLES=4
            shift
            ;;
        --skip-gen)
            EXTRA_ARGS="${EXTRA_ARGS} --skip_generation"
            shift
            ;;
        --include-code-f)
            PROBLEM_TYPES="code_i code_o code_f"
            shift
            ;;
        --tp)
            TENSOR_PARALLEL_SIZE=$2
            shift 2
            ;;
        --gpu-util)
            GPU_MEMORY_UTILIZATION=$2
            shift 2
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# Required for tensor parallelism with vLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Run analysis with vLLM
python analysis/uncertainty_analysis.py \
    --model_path ${MODEL_PATH} \
    --seed_path ${SEED_PATH} \
    --num_tasks ${NUM_TASKS} \
    --num_solver_samples ${NUM_SOLVER_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --problem_types ${PROBLEM_TYPES} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --seed 42 \
    ${EXTRA_ARGS}

echo "Analysis complete! Results saved to ${OUTPUT_DIR}"

