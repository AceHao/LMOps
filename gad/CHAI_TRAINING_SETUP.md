# Chai Opus Training on RunPod - Complete Guide

> **Main entry point for Chai Opus training. Follow this guide from start to finish.**

## 🚀 Quick Start

After completing setup (see below), start training with ONE command:

### For 3B Model (Recommended for testing):
```bash
cd verl && git checkout warmup && cd ..
bash scripts/train/chai-3b-warmup-lr1e-6.sh \
  --model /tmp/Qwen2.5-3B-Instruct \
  --reward_model /tmp/Qwen2.5-3B-Instruct \
  --exp_name chai-3b-warmup \
  --nnodes 1
```

### For 7B Model:
```bash
cd verl && git checkout warmup && cd ..
bash scripts/train/chai-7b-warmup-lr1e-6.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --reward_model /tmp/Qwen2.5-7B-Instruct \
  --exp_name chai-7b-warmup \
  --nnodes 1
```

### For 14B Model:
```bash
cd verl && git checkout warmup && cd ..
bash scripts/train/chai-14b-warmup-lr1e-6.sh \
  --model /tmp/Qwen2.5-14B-Instruct \
  --reward_model /tmp/Qwen2.5-14B-Instruct \
  --exp_name chai-14b-warmup \
  --nnodes 1
```

### For 235B Model:
```bash
cd verl && git checkout warmup && cd ..
bash scripts/train/chai-235b-warmup-lr1e-6.sh \
  --model /tmp/Qwen2.5-235B-Instruct \
  --reward_model /tmp/Qwen2.5-235B-Instruct \
  --exp_name chai-235b-warmup \
  --nnodes 1
```

**What this does:**
- Runs warmup training (2 epochs, ~2,204 steps)
- Saves checkpoints every 100 steps to `/tmp/{exp_name}/`
- Logs metrics to WANDB

**After warmup completes:** See "Adversarial Training" section below

---

## Prerequisites

Before starting, ensure you have:
- [ ] RunPod account with 8x B100 GPU pod available
- [ ] SSH client installed locally
- [ ] WANDB account and API key ([sign up here](https://wandb.ai/))
- [ ] Chai Opus parquet files available locally:
  - `transformed_chai_train.parquet` (675MB, 282K records)
  - `transformed_chai_val.parquet` (35MB, 14K records)

---

## Setup Process (Automated)

For detailed RunPod deployment instructions, see [CHAI_RUNPOD_SETUP.md](CHAI_RUNPOD_SETUP.md).


---

## Training Execution

### Warmup Training

After setup completes, start warmup training using the commands from the Quick Start section above.

**Monitoring:**
- View live metrics in WANDB: `https://wandb.ai/<your-project>`
- Checkpoints saved to: `/tmp/{exp_name}/global_step_*/`
- Validation runs every 100 steps

**Warmup completes at:** ~step 2,204 (2 epochs)

### Adversarial Training

After warmup completes (~step 2,204), prepare and run adversarial training:

#### For 3B Model:
```bash
# Switch verl to gad branch
cd verl && git checkout gad && cd ..

# Prepare adversarial training directory
STEP=2204
mkdir /tmp/chai-3b-adversarial
cp -r /tmp/chai-3b-warmup/global_step_${STEP} /tmp/chai-3b-adversarial/
echo ${STEP} > /tmp/chai-3b-adversarial/latest_checkpointed_iteration.txt

# Run adversarial training
bash scripts/train/chai-3b-adversarial-lr1e-6.sh \
  --exp_name chai-3b-adversarial \
  --resume_step 2204 \
  --nnodes 1
```

#### For 7B Model:
```bash
# Switch verl to gad branch
cd verl && git checkout gad && cd ..

# Prepare adversarial training directory
STEP=2204
mkdir /tmp/chai-7b-adversarial
cp -r /tmp/chai-7b-warmup/global_step_${STEP} /tmp/chai-7b-adversarial/
echo ${STEP} > /tmp/chai-7b-adversarial/latest_checkpointed_iteration.txt

# Run adversarial training
bash scripts/train/chai-7b-adversarial-lr1e-6.sh \
  --exp_name chai-7b-adversarial \
  --resume_step 2204 \
  --nnodes 1
```

#### For 14B Model:
```bash
# Switch verl to gad branch
cd verl && git checkout gad && cd ..

# Prepare adversarial training directory
STEP=2204
mkdir /tmp/chai-14b-adversarial
cp -r /tmp/chai-14b-warmup/global_step_${STEP} /tmp/chai-14b-adversarial/
echo ${STEP} > /tmp/chai-14b-adversarial/latest_checkpointed_iteration.txt

# Run adversarial training
bash scripts/train/chai-14b-adversarial-lr1e-6.sh \
  --exp_name chai-14b-adversarial \
  --resume_step 2204 \
  --nnodes 1
```

#### For 235B Model:
```bash
# Switch verl to gad branch
cd verl && git checkout gad && cd ..

# Prepare adversarial training directory
STEP=2204
mkdir /tmp/chai-235b-adversarial
cp -r /tmp/chai-235b-warmup/global_step_${STEP} /tmp/chai-235b-adversarial/
echo ${STEP} > /tmp/chai-235b-adversarial/latest_checkpointed_iteration.txt

# Run adversarial training
bash scripts/train/chai-235b-adversarial-lr1e-6.sh \
  --exp_name chai-235b-adversarial \
  --resume_step 2204 \
  --nnodes 1
```

**Important:**
- **CRITICAL**: Always switch verl branches before training (`cd verl && git checkout <branch> && cd ..`)
  - Warmup training requires `warmup` branch
  - Adversarial training requires `gad` branch
  - The algorithm implementation differs between branches
- Adversarial training uses a **different** `--exp_name` than warmup (e.g., `chai-3b-adversarial` vs `chai-3b-warmup`)
- The warmup checkpoint is copied to a new directory before starting adversarial training
- Adjust `--resume_step` to match your actual warmup completion step (check `/tmp/chai-*-warmup/` for latest checkpoint)

**Adversarial training runs for:** 4 epochs (~4,408 steps)

---

## Model Size Selection

All training scripts are optimized for **8x B100 GPUs**.

| Model Size | GPU Count | Memory/GPU | Training Speed | Quality | Use Case |
|------------|-----------|------------|----------------|---------|----------|
| **3B** | 8 | ~32GB | Fastest | Good | Testing, prototyping |
| **7B** | 8 | ~32GB | Fast | Better | Development |
| **14B** | 8 | ~40GB | Moderate | Great | Production |
| **235B** | 8+ | ~80GB | Slow | Best | High-quality production |

### Key Parameter Differences by Model Size

| Parameter | 3B | 7B | 14B | 235B | Notes |
|-----------|----|----|-----|------|-------|
| Tensor Parallelism | 2 | 2 | 4 | 8 | 3B/7B use proven settings |
| Tokens/GPU | 12,288 | 12,288 | 8,192 | 4,096 | 3B/7B match reference |
| Train Batch Size | 256 | 256 | 128 | 64 | 3B/7B conservative |
| Val Batch Size | 600 | 600 | 300 | 150 | 3B/7B match reference |
| GPU Memory Util | 0.7 | 0.7 | 0.8 | 0.9 | 3B/7B match reference |
| PPO Mini Batch | 256 | 256 | 128 | 64 | Matches train batch |

**Configuration Strategy:**
- **3B & 7B**: Use identical settings from proven reference 7B script for minimal risk
- **14B & 235B**: Scaled parameters to accommodate larger model sizes
- All configurations validated with 8x B100 GPUs

**Recommendations:**
- **3B**: Start here for testing the pipeline (fastest, same proven config as 7B)
- **7B**: Recommended for most use cases (good quality, proven config)
- **14B**: Use when quality is more important than speed
- **235B**: Maximum quality, requires significant compute budget

---

## Monitoring Training

### WANDB Dashboard

Monitor real-time metrics at: `https://wandb.ai/<your-project>`

**Key metrics to watch:**
- **Loss curves**: actor_loss, critic_loss
- **KL divergence**: Should stay stable around 0.001
- **Rewards**: Validation rewards should improve
- **Learning rates**: Verify they match 1e-6
- **Gradient norms**: Watch for exploding/vanishing gradients

### Checkpoint Verification

List warmup checkpoints:
```bash
ls -lh /tmp/chai-3b-warmup/global_step_*/
```

List adversarial checkpoints:
```bash
ls -lh /tmp/chai-3b-adversarial/global_step_*/
```

Check specific checkpoint:
```bash
ls -lh /tmp/chai-3b-warmup/global_step_2204/actor/
ls -lh /tmp/chai-3b-warmup/global_step_2204/critic/
```

### Training Progress

**Warmup stage (2 epochs):**
- Total steps: ~2,204
- Validation every: 100 steps
- Checkpoint every: 100 steps
- Expected checkpoints: ~22

**Adversarial stage (4 epochs):**
- Total steps: ~4,408
- Validation every: 100 steps
- Checkpoint every: 100 steps
- Expected checkpoints: ~44

---

## Training Pipeline Overview

The GAD (Generative Adversarial Distillation) training consists of two stages:

### Stage 1: Warmup Training
- **Purpose**: Initialize the critic/discriminator model
- **Duration**: 2 epochs = 2,204 steps
- **Learning rate**: 1e-6
- **Critic warmup**: 10 steps
- **Output**: Warmed-up actor and critic checkpoints

### Stage 2: Adversarial Training
- **Purpose**: Full adversarial training for distillation
- **Duration**: 4 epochs = 4,408 steps
- **Learning rate**: 1e-6
- **Critic warmup**: 0 (already warmed)
- **Output**: Distilled model checkpoints

```
┌─────────────────────────────────────────┐
│  Warmup Training (2 epochs, 2,204 steps)│
│  Initialize actor + critic              │
└─────────────────────────────────────────┘
              ↓
   Checkpoint at step 2,204
   /tmp/chai-3b-warmup/global_step_2204/
              ↓
   Copy checkpoint to new directory
   mkdir /tmp/chai-3b-adversarial
   cp -r checkpoint + create iteration file
              ↓
┌─────────────────────────────────────────┐
│  Adversarial Training (4 epochs, 4,408) │
│  Full distillation training             │
└─────────────────────────────────────────┘
              ↓
   Checkpoints every 100 steps
   /tmp/chai-3b-adversarial/global_step_*/
```

---

## Troubleshooting

### "Parquet files not found"
**Symptom:** Setup script exits with error about missing parquet files.

**Solution:**
```bash
# Verify files were SCPd correctly
ls -lh /tmp/LMOps/gad/chai_opus_data/

# If missing, re-run SCP commands from your local machine
scp gad/chai_opus_data/*.parquet root@<runpod-ip>:/tmp/LMOps/gad/chai_opus_data/
```

### "WANDB authentication failed"
**Symptom:** Setup script can't authenticate with WANDB.

**Solution:**
```bash
# Manually login to WANDB
wandb login

# Enter your API key when prompted
# Get API key from: https://wandb.ai/authorize
```

### "Model download failed"
**Symptom:** HuggingFace model download times out or fails.

**Solution:**
```bash
# Check internet connection
ping huggingface.co

# Manually download model
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /tmp/Qwen2.5-3B-Instruct

# Or use environment variable for auth if needed
export HF_TOKEN=<your-hf-token>
```

### "GPU out of memory"
**Symptom:** Training crashes with CUDA OOM error.

**Solution:**
- Use smaller model size (e.g., 3B instead of 7B)
- Or reduce batch size (edit training script):
  - Change `data.train_batch_size=256` to `data.train_batch_size=128`
  - Change `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288` to `8192`

### "Checkpoint not found" (adversarial stage)
**Symptom:** Adversarial script can't find warmup checkpoint.

**Solution:**
```bash
# List available warmup checkpoints
ls -lh /tmp/chai-3b-warmup/global_step_*/

# Switch to gad branch
cd verl && git checkout gad && cd ..

# Use the actual step number when preparing adversarial directory
# Example: If last checkpoint is global_step_2150, use STEP=2150
STEP=2150
mkdir /tmp/chai-3b-adversarial
cp -r /tmp/chai-3b-warmup/global_step_${STEP} /tmp/chai-3b-adversarial/
echo ${STEP} > /tmp/chai-3b-adversarial/latest_checkpointed_iteration.txt

# Run adversarial training with matching resume_step
bash scripts/train/chai-3b-adversarial-lr1e-6.sh \
  --exp_name chai-3b-adversarial \
  --resume_step 2150 \
  --nnodes 1
```

### "Training is slow"
**Symptom:** Training progresses slower than expected.

**Possible causes:**
- Using larger model than GPU capacity
- Network bottleneck (data loading)
- WANDB logging overhead

**Solutions:**
- Verify GPU utilization: `nvidia-smi`
- Check if GPUs are fully utilized
- Consider disabling WANDB temporarily to test (comment out `trainer.logger=['console','wandb']`)

---

## Data Format

The Chai Opus data has been transformed to GAD format:

```json
{
  "content": [
    {"role": "system", "content": "You are Felix Fathom..."},
    {"role": "user", "content": "i walk into class..."},
    {"role": "assistant", "content": "*The class grew quiet...*"},
    {"role": "user", "content": "i smile at everyone"}
  ],
  "teacher_response": "*Miss Bustier smiled warmly.* \"Class, this is Sara...\""
}
```

**Key fields:**
- `content`: Conversation history (list of messages with role and content)
- `teacher_response`: Target response to distill (string)

The training scripts use `data.prompt_key=content` to specify which field contains the conversation history.

---

## Configuration Details

### Data Configuration
```bash
data.prompt_key=content                           # Conversation history field
data.train_files=/tmp/LMOps/gad/chai_opus_data/transformed_chai_train.parquet
data.val_files=/tmp/LMOps/gad/chai_opus_data/transformed_chai_val.parquet
data.train_batch_size=256                         # Training batch size (varies by model)
data.val_batch_size=600                           # Validation batch size (varies by model)
data.max_prompt_length=2048                       # Max conversation tokens
data.max_response_length=1536                     # Max generated response tokens
data.truncation=right                             # Truncate from right if too long
```

### Training Configuration (Warmup)
- Learning rate: 1e-6 (both actor and critic)
- KL loss coefficient: 0.001
- Critic warmup: 10 steps
- Total epochs: 2
- Save frequency: Every 100 steps
- Validation frequency: Every 100 steps

### Training Configuration (Adversarial)
- Learning rate: 1e-6 (both actor and critic)
- KL loss coefficient: 0.001
- Critic warmup: 0 (already warmed)
- Total epochs: 4
- Save frequency: Every 100 steps
- Validation frequency: Every 100 steps

---

## Reference Documentation

For more detailed information, see:

- **[CHAI_RUNPOD_SETUP.md](CHAI_RUNPOD_SETUP.md)** - Manual RunPod deployment steps (automated by setup script)
- **[WANDB_SETUP.md](WANDB_SETUP.md)** - WANDB environment variable configuration details
- **[TRAINING_SETUP.md](TRAINING_SETUP.md)** - Detailed training configuration reference
- **[tools/](tools/)** - Data transformation utilities (for reference, not needed for training)

---

## Next Steps After Training

1. **Evaluate the trained model** on your validation set or custom benchmarks
2. **Generate sample outputs** using the trained checkpoints to assess quality
3. **Compare checkpoints** from different training steps to find the best performing model
4. **Continue training** if needed by resuming from any checkpoint
5. **Export final model** for deployment

---

**Ready to train!** Follow the setup process above, then use the Quick Start commands to begin training. Monitor progress in WANDB and wait for checkpoints at regular intervals.

For questions or issues, see the Troubleshooting section or refer to the detailed documentation linked above.
