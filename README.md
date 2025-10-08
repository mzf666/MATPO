<div align="center">

# MATPO: Multi-Agent Tool-Integrated Policy Optimization

Train Multiple Agent Roles Within a Single LLM via Reinforcement Learning.

[![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon.svg)](https://arxiv.org/pdf/2510.04678)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Code](https://img.shields.io/badge/code-GitHub-black.svg)](https://github.com/mzf666/MATPO)


</div>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/main_gaia.png" width="220px" alt="GAIA Results"><br>
        <em>GAIA Results</em>
      </td>
      <td align="center">
        <img src="assets/main_frameqa.png" width="220px" alt="FRAMES Results"><br>
        <em>FRAMES Results</em>
      </td>
      <td align="center">
        <img src="assets/main_webwalkerqa.png" width="220px" alt="WebWalkerQA Results"><br>
        <em>WebWalkerQA Results</em>
      </td>
    </tr>
  </table>
</div>

<p align="center">
  <img src="assets/multi_agent_framework.png" width="500px" alt="MATPO Framework">
</p>


<p align="center">
  <em>MATPO allows planner and worker agents to coexist within a single LLM and be trained via RL, achieving an 18.38% relative improvement over single-agent baselines on GAIA-text, FRAMES, and WebWalker-QA.</em>
</p>

## News & Updates

- **[2025-Oct-08]** MATPO-Qwen3-14B checkpoints and rollouts released
- **[2025-Oct-08]** Code and training scripts released
- **[2025-Oct-06]** Arxiv Paper released


## Overview

**MATPO** (Multi-Agent Tool-Integrated Policy Optimization) is a novel reinforcement learning framework that enables training multiple specialized agent roles (planner and worker agents) within a single large language model. 

### The Problem
Current single-agent approaches for multi-turn tool-integrated planning face critical limitations:
- **Context Length Bottleneck**: Tool responses (e.g., web scraping) consume excessive tokens, making long-range planning prohibitive
- **Noisy Tool Responses**: Raw tool responses interfere with the model's attention and planning capabilities

### Our Solution
MATPO introduces a **multi-agent-in-one-model** architecture where:
- A **planner-agent** orchestrates high-level planning and delegates subtasks
- **Worker-agents** handle specific browsing and search tasks with isolated contexts
- Both roles are trained within a **single LLM** using role-specific prompts via reinforcement learning


## Key Features

- **Multi-Agent-in-One-Model**: Train planner and worker agents within a single LLM using role-specific system prompts
- **Principled Credit Assignment**: Extends GRPO with theoretically grounded reward distribution across planner and worker rollouts
- **Easy Integration**: Built on top of [veRL](https://github.com/volcengine/verl), compatible with existing RL training frameworks
- **Robust Training**: More stable learning curves compared to single-agent approaches, especially with noisy tool responses
- **Infrastructure Efficient**: No need for deployment of separate models or additional rollout engines


## MATPO Architecture

MATPO employs a hierarchical multi-agent framework where a single LLM serves multiple roles:

```
User Query → Planner Agent → Subtask 1 → Worker Agent → Result 1
                           → Subtask 2 → Worker Agent → Result 2
                           → ...
                           → Final Answer
```


<p align="center">
  <img src="assets/single_agent.png" width="600px" alt="Single-agent GRPO Framework">
  <img src="assets/multi_agent_RL_rollout.png" width="600px" alt="MATPO Framework">
</p>

<p align="center">
  <em>Comparison between the rollout trajectories between the single-agent GRPO (top) and the multi-agent MATPO (bottom).</em>
</p>


### Multi-Agent Rollout Process

1. **Planner Agent**: 
   - Receives user query with planner-specific system prompt
   - Generates high-level plan and decomposes it into subtasks
   - Delegates subtasks to worker agents
   - Synthesizes worker responses into final answer

2. **Worker Agent**:
   - Receives subtask with worker-specific system prompt
   - Performs multi-turn tool-integrated planning (search, scrape, analyze)
   - Returns summarized result to planner
   - Maintains isolated context to prevent token overflow

3. **Credit Assignment**:
   - Final answer accuracy determines the reward
   - Reward is normalized across all planner-worker rollout groups
   - Gradient flows to both planner actions and worker actions proportionally

 
<p align="center">
  <img src="assets/multi-agent-grpo-implementation.png" width="600px" alt="MATPO Framework">
</p>

<p align="center">
  <em>Visualization of MATPO implementation.</em>
</p>



## Quick Start

Prerequisites:
- Python 3.10 or higher
- CUDA 12.4+ (for GPU support)
- 16 x (8 x 80G-A800) GPUs (for training with Qwen3-14B-base)

Clone the repository.
```bash
git clone https://github.com/mzf666/MATPO.git
cd MATPO
```

For prerequisites installation (CUDA, cuDNN, Apex), we recommend following the [verl prerequisites guide](https://verl.readthedocs.io/en/latest/start/install.html#pre-requisites) which provides detailed instructions for:

- CUDA: Version >= 12.4
- cuDNN: Version >= 9.8.0
- Apex

Setup environment and install dependencies.
```bash
conda create -n matpo python==3.10 -y
conda activate matpo
bash examples/sglang_multiturn/install.sh
```

Setup Node.js for Serper API support. 

MCP (Model Context Protocol) requires Node.js to run MCP servers. Node.js version 18+ is recommended for optimal compatibility with MCP tools.
```bash
target_path=YOUR_TARGET_PATH

# Download Node.js binary (example for Linux x64)
wget https://nodejs.org/dist/v24.2.0/node-v24.2.0-linux-x64.tar.xz

# Extract to your target path
tar -xf node-v24.2.0-linux-x64.tar.xz -C $target_path

# Add to PATH
export NODEJS_HOME=$target_path/node-v24.2.0-linux-x64
export PATH=$NODEJS_HOME/bin:$PATH
export NODE_SHARED=$target_path/node-shared/node_modules
export PATH=$NODE_SHARED/.bin:$PATH

# Verify installation
node --version
npm --version

# Install serper mcp server
mkdir -p $target_path/node-shared
cd $target_path/node-shared
npm init -y
npm install serper-search-scrape-mcp-server
```

Configure the Node.js paths and HTTP / HTTPS proxies (if necessary) in the `examples/sglang_multiturn/launch.sh` script properly.

Download the training and testing datasets to the `data` directory. The prerpocessed datasets can be downloaded [here](https://huggingface.co/datasets/veggiebird/MATPO-data).


Train a Qwen3-14B-base model with MATPO on the MuSiQue dataset and evaluate on the GAIA-text datasets:

```bash
# tested on 16 x (8 x 80G-A800) nodes

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
```

## Experiments and Results

### Main Results

MATPO consistently outperforms single-agent GRPO baselines across all benchmarks:

| Method | GAIA-text | WebWalkerQA | FRAMES | Relative Average Improvement |
|--------|-----------|-------------|---------|---------------------|
| Single-Agent GRPO | 32.16% | 30.14% | 56.22% | - |
| **MATPO (Ours)** | **42.60%** | **33.00%** | **63.64%** | **+18.38%** |

### Training Configuration

- **Base Model**: Qwen3-14B-base
- **Training Dataset**: Filtered MuSiQue dataset.
- **Training Steps**: 180 steps
- **Rollouts per Query**: 8 (for group normalization)
- **Reward Function**: 0.9 × accuracy + 0.1 × tool_format_reward

### Model Checkpoints and Rollouts


We release the trained Qwen3-14B-base model checkpoints at the 180th training step of both [single-agent GRPO](https://huggingface.co/veggiebird/MATPO-single-agent-14b) and [MATPO](https://huggingface.co/veggiebird/MATPO-14b).

The associated model rollouts across various training steps can be found [here](https://huggingface.co/datasets/veggiebird/MATPO-rollout).


### Key Findings

- **More Stable Training**: MATPO exhibits more stable learning curves and avoids catastrophic performance drops observed in single-agent training

- **Robustness to Noise**: Multi-agent decomposition effectively isolates noisy tool responses, preventing them from interfering with high-level planning

- **Better Credit Assignment**: Principled reward distribution across planner and worker rollouts leads to more effective learning


### Practical Implementation Tips

Based on our experiments, we recommend:

- **Final Summary**: Final summaries from worker agents are critical for clean planner-worker interfaces
- **Query Recap**: Recapping original user query in worker prompt significantly improves performance
- **URL Blocking**: Remember to blocking HuggingFace search results to avoid data leakage

## Citation

If you find MATPO helpful in your research, please consider citing our paper:

```bibtex
@misc{mo2025multiagenttoolintegratedpolicyoptimization,
      title={Multi-Agent Tool-Integrated Policy Optimization}, 
      author={Zhanfeng Mo and Xingxuan Li and Yuntao Chen and Lidong Bing},
      year={2025},
      eprint={2510.04678},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.04678}, 
}
```


## Acknowledgments

We would like to thank:

- **VolcEngine** for developing and open-sourcing [veRL](https://github.com/volcengine/verl), the RL training framework that powers MATPO
- **Alibaba Cloud** for the Qwen3 model series
- **Google** for the Serper API that enables web search capabilities
- The authors of **GAIA**, **WebWalkerQA**, **FRAMES**, and **MuSiQue** datasets
- The open-source community for valuable feedback and contributions


## FAQ

<details>
<summary><b>Q: What's the difference between MATPO and traditional multi-agent systems?</b></summary>

MATPO uses a single LLM to play multiple agent roles via different system prompts, rather than deploying separate models. This offers:
- Lower infrastructure complexity
- Better parameter efficiency
- Easier deployment and maintenance
- Compatible with existing RL frameworks
</details>

<details>
<summary><b>Q: Can I use MATPO with models other than Qwen3?</b></summary>

Yes! MATPO is model-agnostic. You can use any decoder-only LLM that supports tool calling and multi-turn conversations. We've tested with Qwen3-14B-base, but models like Llama 3, Mistral, or other reasoning-capable LLMs should work.
</details>

<details>
<summary><b>Q: How many GPUs do I need for training?</b></summary>

For Qwen3-14B-base, we recommend:
- **Training**: 8x A100/A800 GPUs (80GB)
- **Inference**: 1-2x A100/A800 GPUs (40GB/80GB)

</details>

<details>
<summary><b>Q: How does MATPO handle credit assignment?</b></summary>

MATPO extends GRPO with principled credit assignment:
1. The planner's final answer determines the accuracy reward
2. This reward is normalized across all rollouts in a group
3. Gradients flow proportionally to both planner and worker actions
4. Worker agents receive the same advantage value as their parent planner rollout

See our paper for more details.
</details>

<details>
<summary><b>Q: Can I use MATPO for tasks other than web search?</b></summary>

Absolutely! While our paper focuses on web search, MATPO's framework is general. You can extend it to:
- Code generation with execution feedback
- Scientific reasoning with calculator tools
- Data analysis with pandas/SQL tools
- Any multi-turn task with verifiable rewards
</details>

<details>
<summary><b>Q: How stable is MATPO training compared to single-agent RL?</b></summary>

MATPO is significantly more stable. Our experiments show:
- Single-agent GRPO often suffers catastrophic drops after step 120
- MATPO maintains steady improvement throughout training
- Multi-agent structure isolates noisy tool responses, preventing interference

See Figure 4 in our paper for training curves.
</details>

<details>
<summary><b>Q: Do I need to block HuggingFace URLs during training?</b></summary>

For research integrity, yes - especially if your evaluation benchmarks are hosted on HuggingFace. This prevents models from "cheating" by finding ground-truth answers online. 

For production systems with no data leakage concerns, this is optional.
</details>

-----

<p align="center">
  <strong>Star ⭐ this repository if you find it helpful!</strong>
</p>
