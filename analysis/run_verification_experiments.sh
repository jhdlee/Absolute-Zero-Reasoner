#!/bin/bash
# Run Verification Mode Comparison Experiments
#
# This script runs training experiments with three verification strategies:
# 1. Full Execution (ground truth baseline)
# 2. Full LLM (LLM-as-a-judge)
# 3. Adaptive (uncertainty-guided, our method)
#
# Usage:
#   ./run_verification_experiments.sh                    # Run with defaults
#   ./run_verification_experiments.sh --num_steps 50     # More steps
#   ./run_verification_experiments.sh --quick            # Quick test run

set -e  # Exit on error

# ==============================================================================
# Configuration
# ==============================================================================

NUM_STEPS=${NUM_STEPS:-10}
TASKS_PER_STEP=${TASKS_PER_STEP:-20}
PROBLEM_TYPES=${PROBLEM_TYPES:-"code_i code_o"}
BUDGET=${BUDGET:-0.3}
N_SAMPLES_UQ=${N_SAMPLES_UQ:-8}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"verification_experiments"}
COMPARISON_DIR=${COMPARISON_DIR:-"verification_comparison"}

# Flags
RUN_ALL=true
RUN_EXECUTION=false
RUN_LLM=false
RUN_ADAPTIVE=false
SKIP_COMPARISON=false

# ==============================================================================
# Parse Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --tasks_per_step)
            TASKS_PER_STEP="$2"
            shift 2
            ;;
        --problem_types)
            PROBLEM_TYPES="$2"
            shift 2
            ;;
        --budget)
            BUDGET="$2"
            shift 2
            ;;
        --n_samples_uq)
            N_SAMPLES_UQ="$2"
            shift 2
            ;;
        --tensor_parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --comparison_dir)
            COMPARISON_DIR="$2"
            shift 2
            ;;
        --only_execution)
            RUN_ALL=false
            RUN_EXECUTION=true
            shift
            ;;
        --only_llm)
            RUN_ALL=false
            RUN_LLM=true
            shift
            ;;
        --only_adaptive)
            RUN_ALL=false
            RUN_ADAPTIVE=true
            shift
            ;;
        --skip_comparison)
            SKIP_COMPARISON=true
            shift
            ;;
        --quick)
            NUM_STEPS=3
            TASKS_PER_STEP=5
            N_SAMPLES_UQ=4
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num_steps N          Number of training steps (default: 10)"
            echo "  --tasks_per_step N     Tasks per step (default: 20)"
            echo "  --problem_types TYPES  Problem types (default: 'code_i code_o')"
            echo "  --budget F             Budget fraction for adaptive (default: 0.3)"
            echo "  --n_samples_uq N       Samples for UQ in adaptive (default: 8)"
            echo "  --tensor_parallel N    VLLM tensor parallel size (default: 1)"
            echo "  --output_dir DIR       Output directory (default: verification_experiments)"
            echo "  --comparison_dir DIR   Comparison output directory (default: verification_comparison)"
            echo "  --only_execution       Only run full execution mode"
            echo "  --only_llm             Only run full LLM mode"
            echo "  --only_adaptive        Only run adaptive mode"
            echo "  --skip_comparison      Skip comparison analysis"
            echo "  --quick                Quick test run (3 steps, 5 tasks)"
            echo "  --help                 Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Setup
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "================================================================================"
echo "VERIFICATION MODE COMPARISON EXPERIMENTS"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Comparison directory: $COMPARISON_DIR"
echo "  Number of steps: $NUM_STEPS"
echo "  Tasks per step: $TASKS_PER_STEP"
echo "  Problem types: $PROBLEM_TYPES"
echo "  Adaptive budget: $BUDGET"
echo "  UQ samples: $N_SAMPLES_UQ"
echo "  Tensor parallel: $TENSOR_PARALLEL"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p "$COMPARISON_DIR"

# ==============================================================================
# Helper Function
# ==============================================================================

run_mode() {
    local mode=$1
    echo ""
    echo "================================================================================"
    echo "Running: $mode"
    echo "================================================================================"

    python -m analysis.train_with_verification \
        --mode "$mode" \
        --num_steps "$NUM_STEPS" \
        --tasks_per_step "$TASKS_PER_STEP" \
        --problem_types $PROBLEM_TYPES \
        --budget "$BUDGET" \
        --n_samples_uq "$N_SAMPLES_UQ" \
        --output_dir "$OUTPUT_DIR" \
        --tensor_parallel_size "$TENSOR_PARALLEL"

    echo "Completed: $mode"
}

# ==============================================================================
# Run Experiments
# ==============================================================================

START_TIME=$(date +%s)

if [ "$RUN_ALL" = true ]; then
    echo "Running all three verification modes..."

    run_mode "full_execution"
    run_mode "full_llm"
    run_mode "adaptive"

elif [ "$RUN_EXECUTION" = true ]; then
    run_mode "full_execution"
elif [ "$RUN_LLM" = true ]; then
    run_mode "full_llm"
elif [ "$RUN_ADAPTIVE" = true ]; then
    run_mode "adaptive"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "================================================================================"
echo "EXPERIMENTS COMPLETED"
echo "================================================================================"
echo "Total time: ${DURATION}s"

# ==============================================================================
# Run Comparison Analysis
# ==============================================================================

if [ "$SKIP_COMPARISON" = false ]; then
    echo ""
    echo "================================================================================"
    echo "Running Comparison Analysis"
    echo "================================================================================"

    python -m analysis.compare_verification_modes \
        --input_dir "$OUTPUT_DIR" \
        --output_dir "$COMPARISON_DIR"

    echo ""
    echo "Comparison complete!"
    echo "  Results: $OUTPUT_DIR/"
    echo "  Analysis: $COMPARISON_DIR/"
fi

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "================================================================================"
echo "EXPERIMENT SUMMARY"
echo "================================================================================"
echo ""
echo "Output files:"
echo "  $OUTPUT_DIR/experiment_full_execution.json"
echo "  $OUTPUT_DIR/experiment_full_llm.json"
echo "  $OUTPUT_DIR/experiment_adaptive.json"
echo ""
echo "  $COMPARISON_DIR/comparison_report.txt"
echo "  $COMPARISON_DIR/all_metrics.json"
echo ""
echo "Plots:"
echo "  $COMPARISON_DIR/reward_comparison.png"
echo "  $COMPARISON_DIR/time_comparison.png"
echo "  $COMPARISON_DIR/accuracy_comparison.png"
echo "  $COMPARISON_DIR/quality_cost_tradeoff.png"
echo ""
echo "To view the report:"
echo "  cat $COMPARISON_DIR/comparison_report.txt"
echo ""
