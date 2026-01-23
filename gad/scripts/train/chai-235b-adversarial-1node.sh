#!/bin/bash
#
# GAD Adversarial Training Script for 235B Models - SINGLE NODE (8 B200 GPUs)
#
# Usage: bash scripts/train/chai-235b-adversarial-1node.sh \
#          --exp_name qwen3-235b-warmup-1n-0122 \
#          --resume_step 100
#
# Note: This script resumes from a warmup checkpoint. The exp_name should match
#       the warmup experiment, and resume_step is the checkpoint step to load.
#

set -x

export NCCL_TIMEOUT=36000
export RAY_memory_usage_threshold=0.98

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --resume_step)
            RESUME_STEP="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

export WANDB_INIT_TIMEOUT=600
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT="gad-replication-qwen3-235b-opus"
export WANDB_API_KEY="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"

export HYDRA_FULL_ERROR=1

# Single-node batch sizes (1 node = 8 GPUs)
TRAIN_BATCH_SIZE=1
VAL_BATCH_SIZE=1
MINI_BATCH_SIZE=8

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAD_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Data and checkpoint directories (local paths)
DATA_DIR="${GAD_DIR}/chai_opus_data"
CHECKPOINT_DIR="/tmp"

# Model paths from warmup checkpoint
model_path="${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/actor/huggingface"
mkdir -p ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/actor/huggingface/
find ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/actor/ -maxdepth 1 -type f ! -name "*.pt" -exec cp {} ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/actor/huggingface/ \;
python tools/merge_model2hf.py --local_dir ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/actor
echo "Files in ${CHECKPOINT_DIR}/$EXP_NAME/global_step_$RESUME_STEP/actor/huggingface:"
ls ${CHECKPOINT_DIR}/$EXP_NAME/global_step_$RESUME_STEP/actor/huggingface

reward_model_path="${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/critic/huggingface"
mkdir -p ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/critic/huggingface/
find ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/critic/ -maxdepth 1 -type f ! -name "*.pt" -exec cp {} ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/critic/huggingface/ \;
python tools/merge_model2hf.py --local_dir ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/critic
echo "Files in ${CHECKPOINT_DIR}/$EXP_NAME/global_step_$RESUME_STEP/critic/huggingface:"
ls ${CHECKPOINT_DIR}/$EXP_NAME/global_step_$RESUME_STEP/critic/huggingface

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.prompt_key=content \
    data.train_files=${DATA_DIR}/transformed_chai_train.parquet \
    data.val_files=${DATA_DIR}/transformed_chai_val.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=1536 \
    data.truncation=right \
    actor_rollout_ref.model.path=$model_path  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.grad_clip=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=$reward_model_path \
    +critic.model.fsdp_config.model_dtype=bf16 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.optim.lr=1e-6 \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=4096 \
    critic.grad_clip=0.2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=4 "${@:1}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${EXP_NAME}
