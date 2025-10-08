#!/bin/bash

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"
pip install "sglang[all]==0.4.6.post5" --no-cache-dir --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python && pip install torch-memory-saver --no-cache-dir
pip install --no-cache-dir "vllm==0.8.5.post1" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata


echo "2. install basic packages"
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff mcp==1.10.1 tenacity

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention"
# Install flash-attn-2.7.4.post1 (cxx11abi=False)
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


echo "4. install FlashInfer"
# Install flashinfer-0.2.7.post1+cu124 (cxx11abi=False)
# 1. Clone the FlashInfer repository:
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
# 2. Make sure you have installed PyTorch with CUDA support. You can check the PyTorch version and CUDA version by running:
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# 3. Install Ninja build system:
pip install ninja
# 4. Install FlashInfer(AOT mode):
cd flashinfer
git checkout v0.2.7.post1
# Set the TORCH_CUDA_ARCH_LIST environment variable to the supported architectures:
# export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a"
# if you are using a100/a800, you can use the following command:
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a"
# Produces AOT kernels in aot-ops/
python -m flashinfer.aot
python -m pip install --no-build-isolation --verbose .


echo "5. May need to fix opencv"
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

echo "6. Fix CUDA and Liger Kernel versions for fused kernels compatibility"
# Downgrade to verl environment versions that work with fused kernels
pip install --no-cache-dir "cuda-python==12.9.0" "cuda-bindings==12.9.0" "cupy-cuda12x==13.4.1"
pip install --no-cache-dir "liger_kernel==0.5.10"

echo "Successfully installed all packages with fused kernels compatibility"
