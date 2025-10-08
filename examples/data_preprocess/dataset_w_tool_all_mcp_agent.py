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
import os

import datasets
from prompt_utils import (
    format_tool_schemas_for_prompt,
    get_system_prompt,
    load_all_tool_schemas_and_kwargs,
)

from verl.utils.hdfs_io import copy, makedirs


def extract_answer(answer_str):
    """Extract the answer from the answer string.

    Args:
        answer_str (str): The raw answer string

    Returns:
        str: The extracted answer
    """
    # For SimpleQA, the answer is already in a clean format
    return answer_str.strip()


def main(args):
    if args.dataset == "simpleqa":
        data_source = "basicv8vc/SimpleQA"
        n_test = 1500
        dataset = datasets.load_dataset(data_source)
        test_dataset = dataset["test"]

        def make_map_fn(split, system_prompt, base_tools_kwargs):
            def process_fn(example, idx):
                question_raw = example.pop("problem")
                question = question_raw + " " + instruction_following
                answer_raw = example.pop("answer")
                answer = extract_answer(answer_raw)

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
                        "index": idx,
                        "answer": answer_raw,
                        "question": question_raw,
                        "metadata": example.pop("metadata"),
                        "need_tools_kwargs": True,
                        "tools_kwargs": tools_kwargs,
                    },
                }

                return data

            return process_fn

    elif args.dataset == "multihoprag":
        data_source = "yixuantt/MultiHopRAG"
        n_test = 500
        dataset = datasets.load_dataset(data_source, "MultiHopRAG")
        test_dataset = dataset["train"]

        def make_map_fn(split, system_prompt, base_tools_kwargs):
            def process_fn(example, idx):
                question_raw = example.pop("query")
                question = question_raw + " " + instruction_following
                answer_raw = example.pop("answer")
                answer = extract_answer(answer_raw)

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
                        "index": idx,
                        "answer": answer_raw,
                        "question": question_raw,
                        "evidence_list": example.pop("evidence_list"),
                        "question_type": example.pop("question_type"),
                        "need_tools_kwargs": True,
                        "tools_kwargs": tools_kwargs,
                    },
                }

                return data

            return process_fn

    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    instruction_following = "You should follow the format instruction in the requestion strictly and wrap the final answer in \\boxed{}."

    tool_schemas, base_tools_kwargs, agent_tools_map = asyncio.run(load_all_tool_schemas_and_kwargs(args.tool_config_path, is_main_agent=True))
    tools_filter = agent_tools_map["main_agent"]
    tool_schemas_text = format_tool_schemas_for_prompt(tool_schemas, tools_filter)
    system_prompt = get_system_prompt(tool_schemas_text, agent_type="main_agent")

    test_dataset = test_dataset.map(
        function=make_map_fn("test", system_prompt, base_tools_kwargs),
        with_indices=True,
    )

    test_dataset = test_dataset.shuffle(seed=args.seed)

    # Convert train_dataset to datasets.Dataset type
    train_dataset = datasets.Dataset.from_dict({k: test_dataset[k][n_test:] for k in test_dataset.features})
    test_dataset = datasets.Dataset.from_dict({k: test_dataset[k][:n_test] for k in test_dataset.features})

    if args.debug:
        train_dataset = train_dataset.select(range(64))
        test_dataset = test_dataset.select(range(64))
        print("Debug mode: using 64 examples for training and testing.")

    print("Training data size: ", len(train_dataset))
    print("Testing data size: ", len(test_dataset))

    local_dir = args.local_dir
    local_dir = os.path.expanduser(local_dir)
    local_dir = os.path.abspath(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
    print(f"Data is saved to {local_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", required=True, choices=["simpleqa", "multihoprag"])
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--tool_config_path",
        default="examples/sglang_multiturn/config/tool_config/mcp_tool_config_full_agent.yaml",
    )
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    if args.local_dir is None:
        args.local_dir = f"~/data/{args.dataset}_dynamic_mcp_agent"

    if args.debug:
        args.local_dir += "_debug"
        if args.hdfs_dir is not None:
            args.hdfs_dir += "_debug"

    main(args)
