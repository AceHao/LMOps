#!/bin/bash
#
# GAD Adversarial Training Script - Qwen2.5-3B (LoRA)
#
# Hardware: 1 Node x 8 NVIDIA B200
# Settings: LoRA Enabled, TP=1 (Data Parallelism), No Offload
#

set -x

export NCCL_TIMEOUT=36000
# export RAY_memory_usage_threshold=0.98
# export TORCH_COMPILE_DISABLE=1


while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --resume_step)
            RESUME_STEP="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --data_subdir)
            DATA_SUBDIR="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Validate required parameters
if [[ -z "${LEARNING_RATE}" ]]; then
    echo "Error: --lr is required"
    exit 1
fi

if [[ -z "${LORA_RANK}" ]]; then
    echo "Error: --rank is required"
    exit 1
fi

if [[ -z "${DATA_SUBDIR}" ]]; then
    echo "Error: --data_subdir is required"
    exit 1
fi

# Alpha is twice the rank
LORA_ALPHA=$((LORA_RANK * 2))

export WANDB_INIT_TIMEOUT=600
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT="gad-replication-qwen25-3b-opus-lora"
export HYDRA_FULL_ERROR=1

# --- 235B Style Path Structure ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAD_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd "$GAD_DIR/../.." && pwd)"
DATA_DIR="${GAD_DIR}/chai_opus_data"
CHECKPOINT_DIR="/workspace/gad-replication/checkpoints"

# --- Merge checkpoint models to HuggingFace format ---
# Use --clean-model-prefix to clean PEFT prefixes while keeping lora_adapter/ separate
model_path="${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/actor/huggingface"
mkdir -p ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/actor/huggingface/
find ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/actor/ -maxdepth 1 -type f ! -name "*.pt" -exec cp {} ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/actor/huggingface/ \;
python tools/merge_model2hf.py --local_dir ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/actor --clean-model-prefix
echo "Files in ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/actor/huggingface:"
ls ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/actor/huggingface

reward_model_path="${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/critic/huggingface"
mkdir -p ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/critic/huggingface/
find ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/critic/ -maxdepth 1 -type f ! -name "*.pt" -exec cp {} ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/critic/huggingface/ \;
python tools/merge_model2hf.py --local_dir ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/critic --clean-model-prefix
echo "Files in ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/critic/huggingface:"
ls ${CHECKPOINT_DIR}/${EXP_NAME}/global_step_${RESUME_STEP}/critic/huggingface

# --- B200 Tuning ---
# TP=1: 3B fits in one GPU. Enables DP=8 (Fastest).
# Batch=2048: Saturates B200 HBM/Compute.
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=1024
MINI_BATCH_SIZE=256

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.prompt_key=content \
    data.train_files=${DATA_DIR}/${DATA_SUBDIR}/train.parquet \
    data.val_files=${DATA_DIR}/${DATA_SUBDIR}/val.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=1536 \
    data.truncation=right \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=$reward_model_path \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.lora_rank=${LORA_RANK} \
    critic.model.lora_alpha=${LORA_ALPHA} \
    critic.model.target_modules=all-linear \
    critic.optim.lr=1e-4 \
    critic.ppo_max_token_len_per_gpu=24576 \
    critic.grad_clip=1.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=4 "${@:1}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${EXP_NAME}
