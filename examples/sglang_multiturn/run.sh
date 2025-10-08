# test on 16 x (8 x 80G-A800) nodes
# lanch for multi-node training

export SERPER_API_KEY="YOUR_SERPER_API_KEY" && \
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY" && \
export WANDB_API_KEY="YOUR_WANDB_API_KEY" && \
export SINGLENODE=true && \
export RAY_DEBUG=legacy && \
export HYDRA_FULL_ERROR=1 && \
source YOUR_CONDA_PATH activate matpo && \
cd YOUR_PROJECT_PATH && \
bash examples/sglang_multiturn/launch.sh \
    examples/sglang_multiturn/qwen3-14b_musique_MATPO.sh