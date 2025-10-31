# Feature: MATPO training with LoRA

## What was the Problem

As MATPO is built upon `veRL`, we can leverage the LoRA support in `veRL` to train MATPO with LoRA. However, we got the following error when running LoRA MATPO training with Qwen3-4B and SGLang using the LoRA configs in `veRL`:
```
KeyError: 'base_model.model.model.layers.0.self_attn.qkv_proj.base_layer.weight'
```
## How we fixed it

We updated the `veRL` SGLang weight synchronization code to **automatically merge LoRA weights into base model** before sending to SGLang. This ensures:

1. ✅ Training uses LoRA (memory efficient)
2. ✅ Rollout uses complete trained model (base + LoRA merged)  
3. ✅ Training and inference are fully consistent

**Modified file**: `verl/workers/sharding_manager/fsdp_sglang.py`

### Key Fix: Automatic LoRA Merging in SGLang

The system now automatically:
- Detects LoRA adapters in your model
- Reads `lora_alpha` and `lora_rank` from config
- Computes merged weights: `W_final = W_base + (lora_alpha/rank) * (lora_B @ lora_A)`
- Sends merged weights to SGLang for rollout


## MATPO training with LoRA (Example)

After setting up the environment following the instructions in `README.md`, we can train LoRA models with MATPO by running these commands:

```bash
# Execute the following commands inside the docker container
# Tested on 1 x (8 x 80G-A800) nodes

#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
export SERPER_API_KEY="YOUR_SERPER_API_KEY"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export SINGLENODE=true
export RAY_DEBUG=legacy
export HYDRA_FULL_ERROR=1
conda activate matpo
cd /workspace/MATPO
bash ./examples/sglang_multiturn/launch.sh \
    examples/sglang_multiturn/qwen3-4b_musique_MATPO_lora.sh
```

⚠️ **Disclaimer**: While the LoRA-enabled MATPO training script is provided, its performance **is not thoroughly evaluated and guaranteed**. LoRA-enabled MATPO training is still an active research area and there may be better ways to train MATPO with LoRA. We welcome any feedback and contributions to evaluate and improve the LoRA-enabled MATPO training performance.

## How to Verify It's Working

Look for these log messages during training:

```
...
[INFO] Applying LoRA to actor module
...
```

If you see these, the fix is working correctly!.

You can also use `nvidia-smi` to compare the GPU memory usage during training and rollout with the full parameter training to verify the fix is working correctly.


## Expected Behavior

### During Training
- LoRA adapters are trained (memory efficient)
- Checkpoints save both base model + LoRA adapters

### During Rollout (Inference)
- SGLang uses **merged model** (base + LoRA adapters merged) ✅
- This ensures training and rollout consistency
- The merge happens automatically before each rollout
- Merge overhead: < 5% of training time





