#!/bin/bash
# GAD Replication - Qwen3-235B - Single Node Fix (LoRA Edition)
#
# DIAGNOSIS: Previous run failed with OOMKilled (System RAM Exhaustion).
# FIX: Enabled LoRA.
# - Full Finetuning requires ~5.4 TB System RAM (Impossible on 1 node).
# - LoRA requires ~1.7 TB System RAM (Feasible).

set -x

export NCCL_TIMEOUT=36000
export RAY_memory_usage_threshold=0.98
export TORCH_COMPILE_DISABLE=1 

# [CRITICAL] Disable buggy B200 allocator
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 
unset PYTORCH_CUDA_ALLOC_CONF

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --reward_model) REWARD_MODEL_PATH="$2"; shift 2 ;;
        --exp_name) EXP_NAME="$2"; shift 2 ;;
        *) break ;;
    esac
done

export WANDB_INIT_TIMEOUT=600
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT="gad-replication-qwen3-235b-opus"
export HYDRA_FULL_ERROR=1

# Kept at 8, assuming LoRA fits. If OOM persists, lower to 4.
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=8
MINI_BATCH_SIZE=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAD_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd "$GAD_DIR/../.." && pwd)"
DATA_DIR="${GAD_DIR}/chai_opus_data"
CHECKPOINT_DIR="${WORKSPACE_DIR}/checkpoints"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_DIR}/models/Qwen3-235B-A22B}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${WORKSPACE_DIR}/models/Qwen3-235B-A22B}"

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
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.grad_clip=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    ++actor_rollout_ref.actor.strategy=fsdp \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=16 \
    +actor_rollout_ref.rollout.kv_cache_dtype=fp8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=$REWARD_MODEL_PATH \
    +critic.model.fsdp_config.model_dtype=bf16 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    +critic.model.fsdp_config.grad_offload=True \
    critic.model.lora_rank=64 \
    critic.model.lora_alpha=32 \
    critic.model.target_modules=all-linear \
    critic.optim.lr=1e-4 \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=4096 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    ++critic.model.strategy=fsdp \
    critic.grad_clip=0.2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=False \
    trainer.critic_warmup=10 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=2 "${@:1}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${EXP_NAME}
