# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the SimpleQA dataset to parquet format
"""

import argparse
import asyncio
import json
import os
import random
from typing import Dict, List

import datasets
from prompt_utils import (
    format_tool_schemas_for_prompt,
    get_system_prompt,
    load_all_tool_schemas_and_kwargs,
)

INSTRUCTION = "You should follow the format instruction in the requestion strictly and wrap the final answer in \\boxed{}."


def load_jsonl_dataset(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    
    return data


def shuffle_dataset(dataset: List[Dict], seed: int):
    random.seed(seed)
    random.shuffle(dataset)
    return dataset


def make_process_fn(data_source, split, system_prompt, base_tools_kwargs):
    def process_fn(example):
        question_raw = example["task_question"]
        question = question_raw + " " + INSTRUCTION
        answer_raw = example["ground_truth"]
        answer = answer_raw.strip()

        # Create tools_kwargs with ground_truth for this example
        tools_kwargs = {}
        for tool_name, _ in base_tools_kwargs.items():
            tools_kwargs[tool_name] = {"create_kwargs": {"ground_truth": answer}}

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            "ability": "qa",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": example["task_id"],
                "answer": answer_raw,
                "question": question_raw,
                "metadata": example["metadata"],
                "need_tools_kwargs": True,
                "tools_kwargs": tools_kwargs,
            },
        }

        return data

    return process_fn


def main(args):

    # 1. Load tool schemas and base tools kwargs
    tool_schemas, base_tools_kwargs, agent_tools_map = asyncio.run(load_all_tool_schemas_and_kwargs(args.tool_config_path, is_main_agent=True))
    tools_filter = agent_tools_map["main_agent"]
    tool_schemas_text = format_tool_schemas_for_prompt(tool_schemas, tools_filter)
    system_prompt = get_system_prompt(tool_schemas_text, agent_type="main_agent")

    # 2. Load source data and process function
    src_data_path = os.path.join(args.src_data_dir, args.dataset, "standardized_data.jsonl")
    raw_dataset = load_jsonl_dataset(src_data_path)

    dataset = []
    for _ in range(args.repeat):
        dataset.extend(raw_dataset.copy())

    dataset = shuffle_dataset(dataset, args.seed)
    process_fn = make_process_fn(args.dataset, "test", system_prompt, base_tools_kwargs)
    dataset = [process_fn(example) for example in dataset]
    dataset = datasets.Dataset.from_list(dataset)

    print(f"Dataset = {args.dataset}, repeat = {args.repeat}, size = {len(dataset)}")
    
    # 3. Save processed dataset
    os.makedirs(args.dst_data_dir, exist_ok=True)
    dst_data_path = os.path.join(args.dst_data_dir, f"{args.dataset}_repeat_{args.repeat}_test.parquet")
    dataset.to_parquet(dst_data_path)
    print(f"Processed dataset saved to {dst_data_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--src_data_dir", type=str, default="/root/data/MiroThinker_Benchmark/data")
    parser.add_argument("--dataset", type=str, choices=["gaia-2023-validation-text-103", "frames", "webwalkerqa"], required=True)
    parser.add_argument("--dst_data_dir", required=True)
    parser.add_argument(
        "--tool_config_path",
        default="examples/sglang_multiturn/config/tool_config/mcp_tool_config_full_agent.yaml",
    )
    args = parser.parse_args()
    main(args)
