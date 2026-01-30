#!/bin/bash
#
# GAD Warmup Training Script for 32B Models (Qwen3 32B)
#
# Model Size: 32B parameters
# Hardware: 8x B200 GPUs (192GB HBM3e each, 8TB/s bandwidth)
# Optimized settings:
# - tensor_model_parallel_size=8 (32B needs TP=8 to fit on single node)
# - ppo_max_token_len_per_gpu=8192 (reduced from 14B due to larger model)
# - gpu_memory_utilization=0.65 (reduced for larger model memory footprint)
# - train_batch_size=128, val_batch_size=300 (reduced from 14B for memory)
#

set -x

export NCCL_TIMEOUT=36000
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --reward_model)
            REWARD_MODEL_PATH="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

export WANDB_INIT_TIMEOUT=600

export TOKENIZERS_PARALLELISM=true
# Use environment variables if set, otherwise use defaults
export WANDB_PROJECT="gad-replication-qwen3-32b-opus"
export WANDB_API_KEY="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"

export HYDRA_FULL_ERROR=1

# --- Relative Path Structure ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAD_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd "$GAD_DIR/../.." && pwd)"
DATA_DIR="${GAD_DIR}/chai_opus_data"
CHECKPOINT_DIR="${WORKSPACE_DIR}/checkpoints"

# Defaults
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_DIR}/models/Qwen3-32B}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${WORKSPACE_DIR}/models/Qwen3-32B}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.prompt_key=content \
    data.train_files=${DATA_DIR}/5pct/train.parquet \
    data.val_files=${DATA_DIR}/5pct/val.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=300 \
    data.max_prompt_length=2048 \
    data.max_response_length=1536 \
    data.truncation=right \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.grad_clip=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=$REWARD_MODEL_PATH \
    critic.optim.lr=1e-6 \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=8192 \
    critic.grad_clip=0.2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=True \
    trainer.critic_warmup=10 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=2 "${@:1}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${EXP_NAME}
