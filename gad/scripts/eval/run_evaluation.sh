#!/bin/bash
# GAD Evaluation Script
#
# This script runs the complete evaluation pipeline:
# 1. Generate responses from trained checkpoints (uses existing generate.sh)
# 2. Score outputs using LLM-as-a-Judge pairwise comparison
#
# Usage:
#   bash run_evaluation.sh --exp_name YOUR_EXP --ckpt 1000 --judge_model gpt-4o
#
# Prerequisites:
#   - Set OPENAI_API_KEY environment variable
#   - Student model checkpoints at /tmp/{exp_name}/global_step_{ckpt}/
#   - Teacher responses file

set -e

# Default values
JUDGE_MODEL="gpt-4o"
WORKERS=4
VAL_DATA="lmsys"
TEACHER_FILE=""
LIMIT=""
RESUME=""
SKIP_GENERATION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --judge_model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --val_data)
            VAL_DATA="$2"
            shift 2
            ;;
        --teacher_file)
            TEACHER_FILE="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --skip_generation)
            SKIP_GENERATION="true"
            shift
            ;;
        --help)
            echo "GAD Evaluation Script"
            echo ""
            echo "Usage: bash run_evaluation.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --exp_name NAME      Experiment name (required)"
            echo "  --ckpt NUMBER        Checkpoint number to evaluate (required)"
            echo "  --judge_model MODEL  Judge model (default: gpt-4o)"
            echo "  --workers N          Number of parallel workers (default: 4)"
            echo "  --val_data NAME      Validation dataset name (default: lmsys)"
            echo "  --teacher_file PATH  Path to teacher responses JSONL"
            echo "  --limit N            Limit number of comparisons"
            echo "  --resume             Resume from existing output file"
            echo "  --skip_generation    Skip generation step (use existing outputs)"
            echo "  --help               Show this help message"
            echo ""
            echo "Example:"
            echo "  bash run_evaluation.sh \\"
            echo "    --exp_name gpt5-chat-filtered-7b-adversarial-lr1e-6 \\"
            echo "    --ckpt 1000 \\"
            echo "    --judge_model gpt-4o"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$EXP_NAME" ]; then
    echo "Error: --exp_name is required"
    exit 1
fi

if [ -z "$CKPT" ]; then
    echo "Error: --ckpt is required"
    exit 1
fi

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY is not set"
    echo "Please set it: export OPENAI_API_KEY=your_key"
fi

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAD_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CKPT_DIR="/tmp/${EXP_NAME}/global_step_${CKPT}"
STUDENT_FILE="${CKPT_DIR}/${VAL_DATA}_generation_results.jsonl"
OUTPUT_FILE="${CKPT_DIR}/${VAL_DATA}_eval_results.json"

echo "========================================"
echo "GAD Evaluation Pipeline"
echo "========================================"
echo "Experiment: $EXP_NAME"
echo "Checkpoint: $CKPT"
echo "Judge Model: $JUDGE_MODEL"
echo "Validation Data: $VAL_DATA"
echo "Student Results: $STUDENT_FILE"
echo "Output: $OUTPUT_FILE"
echo "========================================"

# Step 1: Check/Generate student responses
if [ "$SKIP_GENERATION" != "true" ]; then
    if [ ! -f "$STUDENT_FILE" ]; then
        echo ""
        echo "Step 1: Generating student responses..."
        echo "----------------------------------------"

        # Use existing generation script
        cd "$GAD_DIR"
        bash scripts/generate/generate.sh \
            --exp_name "$EXP_NAME" \
            --val_data "$VAL_DATA" \
            --ckpt_start "$CKPT" \
            --ckpt_end "$CKPT" \
            --ckpt_step 1 \
            --nnodes 1 \
            --ngpus 8

        if [ ! -f "$STUDENT_FILE" ]; then
            echo "Error: Generation failed - student file not found"
            exit 1
        fi
    else
        echo ""
        echo "Step 1: Student responses already exist, skipping generation"
    fi
else
    echo ""
    echo "Step 1: Skipping generation (--skip_generation flag set)"
fi

# Step 2: Verify student file exists
if [ ! -f "$STUDENT_FILE" ]; then
    echo "Error: Student results file not found: $STUDENT_FILE"
    echo "Please run generation first or check the path"
    exit 1
fi

# Step 3: Verify/find teacher file
if [ -z "$TEACHER_FILE" ]; then
    # Try common locations
    POSSIBLE_TEACHER_FILES=(
        "${CKPT_DIR}/teacher_responses.jsonl"
        "/tmp/${EXP_NAME}/teacher_responses.jsonl"
        "/tmp/teacher_${VAL_DATA}_responses.jsonl"
        "${GAD_DIR}/data/teacher_${VAL_DATA}_responses.jsonl"
    )

    for f in "${POSSIBLE_TEACHER_FILES[@]}"; do
        if [ -f "$f" ]; then
            TEACHER_FILE="$f"
            echo "Found teacher file: $TEACHER_FILE"
            break
        fi
    done

    if [ -z "$TEACHER_FILE" ]; then
        echo ""
        echo "Warning: Teacher responses file not found."
        echo "Please provide --teacher_file PATH or place teacher responses at:"
        for f in "${POSSIBLE_TEACHER_FILES[@]}"; do
            echo "  - $f"
        done
        echo ""
        echo "To generate teacher responses, you can:"
        echo "1. Use the same generation script with teacher model"
        echo "2. Use GPT-4o API to generate reference answers"
        exit 1
    fi
fi

if [ ! -f "$TEACHER_FILE" ]; then
    echo "Error: Teacher file not found: $TEACHER_FILE"
    exit 1
fi

echo "Teacher Results: $TEACHER_FILE"

# Step 4: Run pairwise evaluation
echo ""
echo "Step 2: Running LLM-as-a-Judge evaluation..."
echo "----------------------------------------"

cd "$GAD_DIR"

# Build command
CMD="python scripts/eval/evaluate_pairwise.py \
    --student \"$STUDENT_FILE\" \
    --teacher \"$TEACHER_FILE\" \
    --output \"$OUTPUT_FILE\" \
    --judge-model \"$JUDGE_MODEL\" \
    --workers $WORKERS"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD $RESUME"
fi

eval $CMD

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "========================================"
