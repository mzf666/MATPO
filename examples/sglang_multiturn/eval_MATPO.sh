# tested on 2 x (8 x 80G-A800) nodes

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH=$PROJECT_DIR/examples/sglang_multiturn/config
TOOL_CONFIG_PATH=$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/mcp_tool_config_full_agent.yaml

MODEL_CHEKPOINT_PATH=YOUR_MATPO_MODEL_CHECKPOINT_PATH

config_name=simpleqa_multiturn_rm_batch_grpo_agent
project_name=MATPO

BSZ=64
LR=1e-5

DATA_PATH=YOUR_VAL_DATA_PATH
experiment_name=YOUR_EXPERIMENT_NAME


python3 -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$config_name \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$BSZ \
    data.val_batch_size=$BSZ \
    data.max_prompt_length=3072 \
    data.max_response_length=37888 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_CHEKPOINT_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.enable_final_summary=True \
    actor_rollout_ref.rollout.multi_turn.disable_final_summary_for_main_agent=True \
    actor_rollout_ref.rollout.multi_turn.max_summary_length=4096 \
    actor_rollout_ref.rollout.multi_turn.delete_think_from_subagent_tools=True \
    actor_rollout_ref.rollout.multi_turn.enable_tokenization_sanity_check=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${MLP_WORKER_NUM:-1} \
    trainer.save_freq=30 \
    trainer.test_freq=5 \
    trainer.val_only=True \
    trainer.resume_mode=disable \
    trainer.val_before_train=True \
    trainer.rollout_data_dir=$PROJECT_DIR/train_rollout_data/$experiment_name \
    trainer.validation_data_dir=$PROJECT_DIR/val_rollout_data/$experiment_name \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG_PATH \
    trainer.total_epochs=0 $@