#!/bin/bash
# Full Analysis Pipeline for Adaptive Self-Play Reasoning with Uncertainty Quantification
#
# This script runs the complete analysis pipeline:
# 1. Generate tasks and solve with dual labels (execution vs LLM) + UQ metrics
# 2. Analyze which UQ methods predict label disagreement (AUROC)
# 3. Select diverse uncertain tasks for external verification
#
# Usage:
#   ./run_pipeline.sh                    # Run with defaults
#   ./run_pipeline.sh --num_tasks 50     # Override task count
#   ./run_pipeline.sh --skip_generate    # Skip step 1 (use existing results)

set -e  # Exit on error

# ==============================================================================
# Configuration (can be overridden via command line)
# ==============================================================================

NUM_TASKS=${NUM_TASKS:-10}
N_SAMPLES=${N_SAMPLES:-8}  # Number of samples for UQ metrics (must be > 1)
PROBLEM_TYPES=${PROBLEM_TYPES:-"code_i code_o"}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
SEED=${SEED:-42}

# Output paths
OUTPUT_DIR=${OUTPUT_DIR:-"analysis_results"}
TASKS_OUTPUT="${OUTPUT_DIR}/tasks_with_uq.json"
UQ_ANALYSIS_DIR="${OUTPUT_DIR}/uq_analysis"
SELECTED_OUTPUT="${OUTPUT_DIR}/selected_tasks.json"

# Diversity sampling settings
BUDGET=${BUDGET:-20}
UNCERTAINTY_METRIC=${UNCERTAINTY_METRIC:-"sc_entropy"}
SIMILARITY_METHOD=${SIMILARITY_METHOD:-"combined"}
DIVERSITY_METHOD=${DIVERSITY_METHOD:-"mmr"}
LAMBDA_TRADEOFF=${LAMBDA_TRADEOFF:-0.5}

# Flags
SKIP_GENERATE=false
SKIP_ANALYZE=false
SKIP_DIVERSITY=false
SHOW_PROMPTS=false

# ==============================================================================
# Parse command line arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        --n_samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --problem_types)
            PROBLEM_TYPES="$2"
            shift 2
            ;;
        --tensor_parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            TASKS_OUTPUT="${OUTPUT_DIR}/tasks_with_uq.json"
            UQ_ANALYSIS_DIR="${OUTPUT_DIR}/uq_analysis"
            SELECTED_OUTPUT="${OUTPUT_DIR}/selected_tasks.json"
            shift 2
            ;;
        --budget)
            BUDGET="$2"
            shift 2
            ;;
        --uncertainty_metric)
            UNCERTAINTY_METRIC="$2"
            shift 2
            ;;
        --similarity_method)
            SIMILARITY_METHOD="$2"
            shift 2
            ;;
        --diversity_method)
            DIVERSITY_METHOD="$2"
            shift 2
            ;;
        --lambda)
            LAMBDA_TRADEOFF="$2"
            shift 2
            ;;
        --skip_generate)
            SKIP_GENERATE=true
            shift
            ;;
        --skip_analyze)
            SKIP_ANALYZE=true
            shift
            ;;
        --skip_diversity)
            SKIP_DIVERSITY=true
            shift
            ;;
        --show_prompts)
            SHOW_PROMPTS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num_tasks N          Number of tasks per problem type (default: 10)"
            echo "  --n_samples N          Number of samples for UQ metrics (default: 8, must be >1)"
            echo "  --problem_types TYPES  Space-separated problem types (default: 'code_i code_o')"
            echo "  --tensor_parallel N    Tensor parallel size for VLLM (default: 1)"
            echo "  --seed N               Random seed (default: 42)"
            echo "  --output_dir DIR       Output directory (default: analysis_results)"
            echo "  --budget N             Number of tasks to select (default: 20)"
            echo "  --uncertainty_metric M UQ metric to use (default: sc_entropy)"
            echo "  --similarity_method M  Similarity method: structural/lexical/tfidf/embedding/combined"
            echo "  --diversity_method M   Sampling method: greedy/mmr/clustering/dpp/dpp_exact"
            echo "  --lambda F             Uncertainty vs diversity tradeoff 0-1 (default: 0.5)"
            echo "  --skip_generate        Skip task generation (use existing results)"
            echo "  --skip_analyze         Skip UQ analysis"
            echo "  --skip_diversity       Skip diversity sampling"
            echo "  --show_prompts         Show prompts during generation"
            echo "  --help                 Show this help message"
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
echo "ADAPTIVE SELF-PLAY REASONING ANALYSIS PIPELINE"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of tasks: $NUM_TASKS per type"
echo "  N samples for UQ: $N_SAMPLES"
echo "  Problem types: $PROBLEM_TYPES"
echo "  Selection budget: $BUDGET"
echo "  UQ metric: $UNCERTAINTY_METRIC"
echo "  Similarity: $SIMILARITY_METHOD"
echo "  Diversity method: $DIVERSITY_METHOD"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$UQ_ANALYSIS_DIR"

# ==============================================================================
# Step 1: Generate Tasks with Dual Labels and UQ Metrics
# ==============================================================================

if [ "$SKIP_GENERATE" = false ]; then
    echo "================================================================================"
    echo "STEP 1: Generating tasks with dual labels and UQ metrics"
    echo "================================================================================"
    echo ""

    GENERATE_ARGS=(
        --num_tasks "$NUM_TASKS"
        --n_samples "$N_SAMPLES"
        --problem_types $PROBLEM_TYPES
        --seed "$SEED"
        --tensor_parallel_size "$TENSOR_PARALLEL"
        --output_path "$TASKS_OUTPUT"
    )

    if [ "$SHOW_PROMPTS" = true ]; then
        GENERATE_ARGS+=(--show_prompts)
    fi

    python -m analysis.generate_tasks "${GENERATE_ARGS[@]}"

    echo ""
    echo "Tasks saved to: $TASKS_OUTPUT"
else
    echo "================================================================================"
    echo "STEP 1: Skipped (using existing results)"
    echo "================================================================================"
    if [ ! -f "$TASKS_OUTPUT" ]; then
        echo "ERROR: $TASKS_OUTPUT not found. Run without --skip_generate first."
        exit 1
    fi
fi

# ==============================================================================
# Step 2: Analyze UQ Methods (AUROC)
# ==============================================================================

if [ "$SKIP_ANALYZE" = false ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 2: Analyzing UQ methods (AUROC for predicting label disagreement)"
    echo "================================================================================"
    echo ""

    python -m analysis.analyze_uncertainty \
        --input_path "$TASKS_OUTPUT" \
        --output_dir "$UQ_ANALYSIS_DIR"

    echo ""
    echo "Analysis results saved to: $UQ_ANALYSIS_DIR"
else
    echo ""
    echo "================================================================================"
    echo "STEP 2: Skipped"
    echo "================================================================================"
fi

# ==============================================================================
# Step 3: Diversity-Aware Sampling
# ==============================================================================

if [ "$SKIP_DIVERSITY" = false ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 3: Selecting diverse uncertain tasks for external verification"
    echo "================================================================================"
    echo ""

    python -m analysis.diversity_sampling \
        --input_path "$TASKS_OUTPUT" \
        --output_path "$SELECTED_OUTPUT" \
        --budget "$BUDGET" \
        --uncertainty_metric "$UNCERTAINTY_METRIC" \
        --similarity_method "$SIMILARITY_METHOD" \
        --diversity_method "$DIVERSITY_METHOD" \
        --lambda_tradeoff "$LAMBDA_TRADEOFF"

    echo ""
    echo "Selected tasks saved to: $SELECTED_OUTPUT"
else
    echo ""
    echo "================================================================================"
    echo "STEP 3: Skipped"
    echo "================================================================================"
fi

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================================"
echo ""
echo "Output files:"
echo "  Tasks with UQ:     $TASKS_OUTPUT"
echo "  UQ Analysis:       $UQ_ANALYSIS_DIR/"
echo "  Selected Tasks:    $SELECTED_OUTPUT"
echo ""
echo "To view results:"
echo "  cat $TASKS_OUTPUT | python -m json.tool | head -100"
echo "  open $UQ_ANALYSIS_DIR/roc_curves.png"
echo "  cat $SELECTED_OUTPUT | python -m json.tool"
echo ""
