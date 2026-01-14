#!/bin/bash
#
# Chai Opus Training Setup Script for RunPod
#
# This script automates the environment setup for Chai Opus training.
# Run this BEFORE transferring parquet files via SCP.
#
# Prerequisites:
# - RunPod pod with verl Docker image deployed
# - SSH access to RunPod pod
# - LMOps repository cloned to /tmp/LMOps (you're already here!)
#
# Usage:
#   cd /tmp/LMOps/gad
#   bash scripts/setup_chai_training.sh
#

set -e  # Exit on error

echo "================================================"
echo "Chai Opus Training Setup for RunPod"
echo "================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${REPO_ROOT}/chai_opus_data"

# Verify we're in the gad directory by checking for local_setup.sh
if [ ! -f "${REPO_ROOT}/local_setup.sh" ]; then
    echo "❌ Error: Cannot find local_setup.sh in ${REPO_ROOT}"
    echo ""
    echo "This script should be run from the LMOps/gad directory."
    echo "Script location: $SCRIPT_DIR"
    echo ""
    exit 1
fi

# Change to repo root to run setup commands
cd "$REPO_ROOT"

echo "✅ Running from: $REPO_ROOT"
echo ""

# Step 1: Install WANDB
echo "Step 1: Installing WANDB"
echo "------------------------"
if command -v wandb &> /dev/null; then
    echo "✅ WANDB already installed"
else
    pip install wandb -q
    echo "✅ WANDB installed"
fi
echo ""

# Step 2: WANDB Authentication
echo "Step 2: WANDB Authentication"
echo "----------------------------"
if wandb status &> /dev/null; then
    echo "✅ Already logged in to WANDB"
    wandb status
else
    echo "Please log in to WANDB (get API key from https://wandb.ai/authorize):"
    wandb login
fi
echo ""

# Step 3: Configure WANDB Environment
echo "Step 3: Configure WANDB Environment"
echo "------------------------------------"
if [ -z "$WANDB_PROJECT" ]; then
    read -p "Enter WANDB project name: " WANDB_PROJECT
    export WANDB_PROJECT

    # Save to bashrc for persistence
    if ! grep -q "WANDB_PROJECT" ~/.bashrc 2>/dev/null; then
        echo "export WANDB_PROJECT='${WANDB_PROJECT}'" >> ~/.bashrc
    fi
fi

if [ -z "$WANDB_API_KEY" ]; then
    # Try to extract API key from wandb status
    WANDB_API_KEY=$(wandb status 2>/dev/null | grep -oP 'Logged in.*key: \K\w+' || echo "")
    if [ -n "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY
        if ! grep -q "WANDB_API_KEY" ~/.bashrc 2>/dev/null; then
            echo "export WANDB_API_KEY='${WANDB_API_KEY}'" >> ~/.bashrc
        fi
    fi
fi

echo "✅ WANDB project: ${WANDB_PROJECT}"
echo "✅ Environment variables configured"
echo ""

# Step 4: Clone verl Repository
echo "Step 4: Setting Up verl Repository"
echo "-----------------------------------"
if [ ! -d "verl" ]; then
    echo "Cloning verl repository from AceHao (has B100 GPU fixes)..."
    git clone https://github.com/AceHao/verl.git
    echo "✅ verl cloned"
else
    echo "✅ verl already exists"
fi
echo ""

# Step 5: Install Dependencies
echo "Step 5: Installing Dependencies"
echo "--------------------------------"
echo "Running local_setup.sh..."
bash local_setup.sh
echo "✅ Dependencies installed"
echo ""

# Step 6: Download Qwen Model
echo "Step 6: Downloading Qwen Model"
echo "-------------------------------"
echo "Which model size do you want to use?"
echo "  1) 3B (recommended for testing)"
echo "  2) 7B"
echo "  3) 14B"
echo "  4) 235B"
echo "  5) Skip model download"
read -p "Enter choice [1-5]: " MODEL_CHOICE

case $MODEL_CHOICE in
    1) MODEL_SIZE="3B";;
    2) MODEL_SIZE="7B";;
    3) MODEL_SIZE="14B";;
    4) MODEL_SIZE="235B";;
    5)
        echo "Skipping model download"
        MODEL_SIZE=""
        ;;
    *)
        echo "Invalid choice, defaulting to 3B"
        MODEL_SIZE="3B"
        ;;
esac

if [ -n "$MODEL_SIZE" ]; then
    MODEL_PATH="/tmp/Qwen2.5-${MODEL_SIZE}-Instruct"

    if [ -d "$MODEL_PATH" ]; then
        echo "✅ Qwen ${MODEL_SIZE} model already exists at $MODEL_PATH"
    else
        echo "Downloading Qwen ${MODEL_SIZE} model..."
        echo "This may take 10-15 minutes depending on network speed..."
        huggingface-cli download Qwen/Qwen2.5-${MODEL_SIZE}-Instruct --local-dir "$MODEL_PATH"
        echo "✅ Qwen ${MODEL_SIZE} model downloaded"
    fi
fi
echo ""

# Step 7: Create Data Directory
echo "Step 7: Preparing Data Directory"
echo "---------------------------------"
mkdir -p "$DATA_DIR"
echo "✅ Data directory created: $DATA_DIR"
echo ""

# Step 8: Environment Setup Complete
echo "================================================"
echo "✅ Environment Setup Complete!"
echo "================================================"
echo ""
echo "Next Step: Transfer Parquet Files"
echo "----------------------------------"
echo ""
echo "On your LOCAL machine, run:"
echo ""
echo "  # Transfer training data (675MB)"
echo "  scp chai_opus_data/transformed_chai_train.parquet root@<runpod-ip>:${DATA_DIR}/"
echo ""
echo "  # Transfer validation data (35MB)"
echo "  scp chai_opus_data/transformed_chai_val.parquet root@<runpod-ip>:${DATA_DIR}/"
echo ""
echo "After transferring files, verify with:"
echo "  ls -lh ${DATA_DIR}/"
echo ""
echo "Then you can start training!"
echo ""

# Step 9: Verify Prerequisites
echo "Verifying environment prerequisites..."
echo ""

WARNINGS=0

[ -d "verl" ] && echo "✅ verl repository" || { echo "❌ verl repository missing"; WARNINGS=$((WARNINGS+1)); }
wandb status &> /dev/null && echo "✅ WANDB authenticated" || { echo "❌ WANDB not authenticated"; WARNINGS=$((WARNINGS+1)); }
[ -n "$WANDB_PROJECT" ] && echo "✅ WANDB project configured" || { echo "⚠️  WANDB project not set"; }
if [ -n "$MODEL_SIZE" ]; then
    [ -d "$MODEL_PATH" ] && echo "✅ Qwen ${MODEL_SIZE} model" || { echo "❌ Model not downloaded"; WARNINGS=$((WARNINGS+1)); }
fi
[ -d "$DATA_DIR" ] && echo "✅ Data directory created" || { echo "❌ Data directory missing"; WARNINGS=$((WARNINGS+1)); }

echo ""

if [ $WARNINGS -gt 0 ]; then
    echo "⚠️  Setup completed with $WARNINGS warning(s)"
    echo "   Please address the issues above before training"
    echo ""
    exit 1
fi

# Step 10: Print Training Commands (to be used after data transfer)
echo "================================================"
echo "After Data Transfer - Training Commands:"
echo "================================================"
echo ""
echo "Once you've transferred the parquet files, start training with:"
echo ""

if [ -n "$MODEL_SIZE" ]; then
    MODEL_LOWER=$(echo $MODEL_SIZE | tr '[:upper:]' '[:lower:]')
    echo "For ${MODEL_SIZE} model:"
    echo ""
    echo "1. Start warmup training:"
    echo "  bash scripts/train/chai-${MODEL_LOWER}-warmup-lr1e-6.sh \\"
    echo "    --model ${MODEL_PATH} \\"
    echo "    --reward_model ${MODEL_PATH} \\"
    echo "    --exp_name chai-${MODEL_LOWER}-warmup \\"
    echo "    --nnodes 1"
    echo ""
    echo "2. After warmup completes (~step 2204), prepare adversarial training:"
    echo "  STEP=2204"
    echo "  mkdir /tmp/chai-${MODEL_LOWER}-adversarial"
    echo "  cp -r /tmp/chai-${MODEL_LOWER}-warmup/global_step_\${STEP} /tmp/chai-${MODEL_LOWER}-adversarial/"
    echo "  echo \${STEP} > /tmp/chai-${MODEL_LOWER}-adversarial/latest_checkpointed_iteration.txt"
    echo ""
    echo "3. Run adversarial training:"
    echo "  bash scripts/train/chai-${MODEL_LOWER}-adversarial-lr1e-6.sh \\"
    echo "    --exp_name chai-${MODEL_LOWER}-adversarial \\"
    echo "    --resume_step 2204 \\"
    echo "    --nnodes 1"
    echo ""
fi

echo "Quick reference for all model sizes:"
echo ""
echo "3B model (recommended for testing):"
echo "  Warmup:      bash scripts/train/chai-3b-warmup-lr1e-6.sh --model /tmp/Qwen2.5-3B-Instruct --reward_model /tmp/Qwen2.5-3B-Instruct --exp_name chai-3b-warmup --nnodes 1"
echo "  Adversarial: bash scripts/train/chai-3b-adversarial-lr1e-6.sh --exp_name chai-3b-adversarial --resume_step 2204 --nnodes 1"
echo ""
echo "7B model:"
echo "  Warmup:      bash scripts/train/chai-7b-warmup-lr1e-6.sh --model /tmp/Qwen2.5-7B-Instruct --reward_model /tmp/Qwen2.5-7B-Instruct --exp_name chai-7b-warmup --nnodes 1"
echo "  Adversarial: bash scripts/train/chai-7b-adversarial-lr1e-6.sh --exp_name chai-7b-adversarial --resume_step 2204 --nnodes 1"
echo ""
echo "14B model:"
echo "  Warmup:      bash scripts/train/chai-14b-warmup-lr1e-6.sh --model /tmp/Qwen2.5-14B-Instruct --reward_model /tmp/Qwen2.5-14B-Instruct --exp_name chai-14b-warmup --nnodes 1"
echo "  Adversarial: bash scripts/train/chai-14b-adversarial-lr1e-6.sh --exp_name chai-14b-adversarial --resume_step 2204 --nnodes 1"
echo ""
echo "235B model:"
echo "  Warmup:      bash scripts/train/chai-235b-warmup-lr1e-6.sh --model /tmp/Qwen2.5-235B-Instruct --reward_model /tmp/Qwen2.5-235B-Instruct --exp_name chai-235b-warmup --nnodes 1"
echo "  Adversarial: bash scripts/train/chai-235b-adversarial-lr1e-6.sh --exp_name chai-235b-adversarial --resume_step 2204 --nnodes 1"
echo ""
echo "Monitor training:"
echo "  WANDB: https://wandb.ai/${WANDB_PROJECT:-your-project}"
echo "  Warmup checkpoints: /tmp/chai-*-warmup/global_step_*/"
echo "  Adversarial checkpoints: /tmp/chai-*-adversarial/global_step_*/"
echo ""
echo "For more details, see: CHAI_TRAINING_SETUP.md"
echo ""
