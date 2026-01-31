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

# Fix fragmentation issues - DO NOT use expandable_segments with vLLM
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.9"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.prompt_key=content \
    data.train_files=${DATA_DIR}/5pct/train.parquet \
    data.val_files=${DATA_DIR}/5pct/val.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.truncation=right \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-5 \
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
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path=$REWARD_MODEL_PATH \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=8192 \
    critic.grad_clip=0.2 \
    critic.use_dynamic_bsz=True \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.model.fsdp_config.param_offload=False \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.lora_rank=64 \
    critic.model.lora_alpha=128 \
    critic.model.target_modules=all-linear \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.strategy=fsdp \
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
    trainer.total_epochs=2 "${@:1}"