# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# NOTE: This file is modified based on tool.py, which is used for the main agent to compute the reward for the tool-rollouts.
# The main difference is that we integrate the tool_format_reward of subagent-tool-rollouts into the accuracy_reward and tool_format_reward of main-agent-rollouts.
# Therefore, in ray_trainer.py, we do NOT need to handle the subagent-tool-rollouts separately when computing the rewards.


import ast
import asyncio
import io
import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import regex
import torch
from openai import AsyncOpenAI
from tqdm import tqdm

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.llm_judge import SIMPLEQA_EVALUATION_PROMPT, extract_boxed_answer
from verl.workers.reward_manager import register
from verl.workers.reward_manager.tool import (
    ToolRewardManager,
    batch_compute_scores_async,
    close_openai_client,
    create_async_openai_client,
    run_async_computation,
)


@register("subagent_tool")
class SubagentToolRewardManager(ToolRewardManager):
    """The reward manager for subagent-tool data."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", accuracy_reward_weight=1.0, tool_format_reward_weight=0.1, gate_tool_format_reward=False, async_process=False, batch_size=10, max_retry=3) -> None:
        """
        Initialize the ToolRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor data. Defaults to "data_source".
            accuracy_reward_weight: Weight for the accuracy reward component. Defaults to 1.0.
            tool_format_reward_weight: Weight for the tool format reward component. Defaults to 0.1.
            gate_tool_format_reward: Whether to gate tool format reward based on accuracy. Defaults to False.
            async_process: Whether to use async processing for reward computation. Defaults to False.
            batch_size: Number of concurrent requests for async processing. Defaults to 10.
            max_retry: Maximum number of retries for failed async requests. Defaults to 3.
        """
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            accuracy_reward_weight=accuracy_reward_weight,
            tool_format_reward_weight=tool_format_reward_weight,
            gate_tool_format_reward=gate_tool_format_reward,
            async_process=async_process, 
            batch_size=batch_size, 
            max_retry=max_retry,
        )

    def __call__(self, data: DataProto, return_dict=False):
        is_from_subagent_tool_list = data.batch["is_from_subagent_tool"].cpu().numpy()
        if any(is_from_subagent_tool_list):
            return self._call_include_subagent_tool(data, return_dict)
        else:
            return self._call_only_main_agent(data, return_dict)

    # NOTE: this function is the same as the original ToolRewardManager.__call__()
    def _call_only_main_agent(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        prompt_str_list = []
        response_str_list = []
        ground_truth_list = []
        data_source_list = []
        extra_info_list = []
        valid_response_length_list = []

        accuracy_reward_list = []
        tool_format_reward_list = []
        combined_reward_list = []


        ### Get information ready for evaluation
        for i in range(len(data)):
            
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            prompt_str_list.append(prompt_str)
            response_str_list.append(response_str)
            ground_truth_list.append(ground_truth)
            data_source_list.append(data_source)
            extra_info_list.append(extra_info)
            valid_response_length_list.append(valid_response_length)

        ### Compute accuracy reward
        if self.async_process:
            # openai batch api
            start_time = time.time()
            print("Start computing accuracy reward with async api, this may take a while...")
            print(f"Async configuration: batch_size={self.batch_size}, max_retry={self.max_retry}")
            try:
                # Create a fresh client for each batch to avoid connection pool issues
                fresh_client = create_async_openai_client(self.openai_api_key)
                
                # Run async computation using the new async functions
                accuracy_reward_list = run_async_computation(
                    fresh_client,
                    data_source_list,
                    response_str_list,
                    ground_truth_list,
                    extra_info_list,
                    self.batch_size,
                    self.max_retry
                )
                
                end_time = time.time()
                print(f"Async batch processing completed in {end_time - start_time:.2f} seconds")

            except Exception as e:
                # Re-raise the exception instead of using fallback
                print(f"Error in batch api: {e}")
                raise e

        else:
            # if not batch process, we compute reward for each sample separately
            for i in range(len(data)):
                data_source = data_source_list[i]
                response_str = response_str_list[i]
                ground_truth = ground_truth_list[i]
                extra_info = extra_info_list[i]

                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

                if isinstance(score, dict):
                    accuracy_reward = score["score"]
                    # Store the information including original reward
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    accuracy_reward = score

                accuracy_reward_list.append(accuracy_reward)

        ### compute tool reward for each sample
        for i in range(len(data)):
            prompt_str = prompt_str_list[i]
            response_str = response_str_list[i]
            ground_truth = ground_truth_list[i]
            extra_info = extra_info_list[i]

            accuracy_reward = accuracy_reward_list[i]

            # tool reward
            mcp_format_reward = self.check_mcp_format(prompt_str, response_str)

            if self.gate_tool_format_reward:
                # if gate_tool_format_reward is True, the tool format reward will not be given if the final answer is not correct, even the tool format is correct.
                if accuracy_reward != 1.0:
                    mcp_format_reward = 0.0
        
            tool_format_reward_list.append(mcp_format_reward)

            # combine accuracy reward and tool reward
            accuracy_reward = accuracy_reward * self.accuracy_reward_weight
            mcp_format_reward = mcp_format_reward * self.tool_format_reward_weight
            reward = accuracy_reward + mcp_format_reward

            # Store separate rewards for logging
            reward_extra_info["accuracy_reward"].append(accuracy_reward)
            reward_extra_info["tool_format_reward"].append(mcp_format_reward)
            reward_extra_info["combined_reward"].append(reward)

            reward_tensor[i, valid_response_length_list[i] - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[accuracy_reward]", accuracy_reward)
                print("[tool_format_reward]", mcp_format_reward)
                print("[combined_reward]", reward)
                if isinstance(reward, dict):
                    for key, value in reward.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


    # NOTE: this function calculate combined_reward for both main-agent-rollouts and subagent-tool-rollouts.
    # combined_reward = accuracy_reward (main-agent-rollouts) + tool_format_reward (main-agent-rollouts) + tool_format_reward (subagent-tool-rollouts)
    def _call_include_subagent_tool(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # build the main-agent <--> subagent-tool relationship
        is_from_subagent_tool_list = data.batch["is_from_subagent_tool"].cpu().numpy()
        reqs_ids_list = data.non_tensor_batch["reqs_ids"]
        parent_reqs_ids_list = data.non_tensor_batch["parent_reqs_ids"]

        parent_idx_list = np.array([None] * len(data))
        for i in range(len(data)):
            if is_from_subagent_tool_list[i]:
                parent_idx_list[i] = np.where(reqs_ids_list == parent_reqs_ids_list[i])[0][0]
            
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        prompt_str_list = []
        response_str_list = []
        ground_truth_list = []
        data_source_list = []
        extra_info_list = []
        valid_response_length_list = []
        tool_format_reward_list = []


        ### Get information ready for evaluation
        for i in range(len(data)):
            
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            prompt_str_list.append(prompt_str)
            response_str_list.append(response_str)
            ground_truth_list.append(ground_truth)
            data_source_list.append(data_source)
            extra_info_list.append(extra_info)
            valid_response_length_list.append(valid_response_length)


        ### Compute accuracy_reward only for main-agent-rollouts, set accuracy_reward as None for subagent-tool-rollouts

        prompt_str_list = np.array(prompt_str_list)
        data_source_list = np.array(data_source_list)
        response_str_list = np.array(response_str_list)
        ground_truth_list = np.array(ground_truth_list)
        extra_info_list = np.array(extra_info_list)

        prompt_str_list_backup = prompt_str_list.copy()
        data_source_list_backup = data_source_list.copy()
        response_str_list_backup = response_str_list.copy()
        ground_truth_list_backup = ground_truth_list.copy()
        extra_info_list_backup = extra_info_list.copy()

        prompt_str_list = prompt_str_list[~is_from_subagent_tool_list]
        data_source_list = data_source_list[~is_from_subagent_tool_list]
        response_str_list = response_str_list[~is_from_subagent_tool_list]
        ground_truth_list = ground_truth_list[~is_from_subagent_tool_list]
        extra_info_list = extra_info_list[~is_from_subagent_tool_list]


        if self.async_process:
            # openai batch api
            start_time = time.time()
            print("Start computing accuracy reward with async api, this may take a while...")
            print(f"Async configuration: batch_size={self.batch_size}, max_retry={self.max_retry}")
            try:
                # Create a fresh client for each batch to avoid connection pool issues
                fresh_client = create_async_openai_client(self.openai_api_key)
                
                # Run async computation using the new async functions
                accuracy_reward_list = run_async_computation(
                    fresh_client,
                    data_source_list,
                    response_str_list,
                    ground_truth_list,
                    extra_info_list,
                    self.batch_size,
                    self.max_retry
                )
                
                end_time = time.time()
                print(f"Async batch processing completed in {end_time - start_time:.2f} seconds")

            except Exception as e:
                # Re-raise the exception instead of using fallback
                print(f"Error in batch api: {e}")
                raise e

        else:
            # if not batch process, we compute reward for each sample separately
            accuracy_reward_list = []

            for i in range(len(data)):
                data_source = data_source_list[i]
                response_str = response_str_list[i]
                ground_truth = ground_truth_list[i]
                extra_info = extra_info_list[i]

                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

                if isinstance(score, dict):
                    accuracy_reward = score["score"]
                    # Store the information including original reward
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    accuracy_reward = score

                accuracy_reward_list.append(accuracy_reward)


        accuracy_reward_list = np.array(accuracy_reward_list)
        assert len(accuracy_reward_list) == len(data_source_list)
        accuracy_reward_list_all = np.array([0.0] * len(data))
        accuracy_reward_list_all[~is_from_subagent_tool_list] = accuracy_reward_list
        accuracy_reward_list = accuracy_reward_list_all
        del accuracy_reward_list_all

        prompt_str_list = prompt_str_list_backup
        data_source_list = data_source_list_backup
        response_str_list = response_str_list_backup
        ground_truth_list = ground_truth_list_backup
        extra_info_list = extra_info_list_backup
        del prompt_str_list_backup, data_source_list_backup, response_str_list_backup, ground_truth_list_backup, extra_info_list_backup


        # broadcast the accuracy_reward to each subagent-tool-rollout according to the main-agent <--> subagent-tool relationship.
        for i in range(len(data)):
            if is_from_subagent_tool_list[i]:
                accuracy_reward_list[i] = accuracy_reward_list[parent_idx_list[i]]

        ### compute tool_format_reward for each sample, both main-agent-rollouts and subagent-tool-rollouts, but do NOT compute the combined_reward for now.
        for i in range(len(data)):
            prompt_str = prompt_str_list[i]
            response_str = response_str_list[i]
            ground_truth = ground_truth_list[i]
            extra_info = extra_info_list[i]
            accuracy_reward = accuracy_reward_list[i]
            assert accuracy_reward is not None, f"accuracy_reward_list[{i}] == None, which means the accuracy_reward for main-agent-rollout is wrongly or broadcasted yet. is_from_subagent_tool_list[{i}] = {is_from_subagent_tool_list[i]}"

            # tool reward
            mcp_format_reward = self.check_mcp_format(prompt_str, response_str)

            if self.gate_tool_format_reward:
                # if gate_tool_format_reward is True, the tool format reward will not be given if the final answer is not correct, even the tool format is correct.
                if accuracy_reward != 1.0:
                    mcp_format_reward = 0.0

            mcp_format_reward = mcp_format_reward * self.tool_format_reward_weight

            tool_format_reward_list.append(mcp_format_reward)
            reward_extra_info["tool_format_reward"].append(mcp_format_reward)

        tool_format_reward_list = np.array(tool_format_reward_list)  

        # compute combined reward for main-agent-rollouts only. 
        # Adding up the final averaged tool_format_reward, using the main-agent <--> subagent-tool relationship
        combined_reward_list = []

        for i in range(len(data)):
            accuracy_reward = accuracy_reward_list[i]
            accuracy_reward = accuracy_reward * self.accuracy_reward_weight

            reward_extra_info["accuracy_reward"].append(accuracy_reward)

            if is_from_subagent_tool_list[i]:
                combined_reward = 0.0
            else:
                mcp_format_reward_main_agent = tool_format_reward_list[i]
                mcp_format_reward_subagent_tool_list = tool_format_reward_list[parent_idx_list == i]
                mcp_format_reward_subagent_tool = np.mean(mcp_format_reward_subagent_tool_list) if len(mcp_format_reward_subagent_tool_list) > 0 else 0.0

                tool_format_reward = 0.5 * (mcp_format_reward_main_agent + mcp_format_reward_subagent_tool)

                combined_reward = accuracy_reward + tool_format_reward

            combined_reward_list.append(combined_reward)

        combined_reward_list = np.array(combined_reward_list)

        # broadcast the accuracy_reward and the combined_reward to each subagent-tool-rollout according to the main-agent <--> subagent-tool relationship. Do NOT modify the tool_format_reward for each subagent-tool-rollout.

        already_print_data_sources = {}

        for i in range(len(data)):

            if is_from_subagent_tool_list[i]:
                combined_reward = combined_reward_list[parent_idx_list[i]]
            else:
                combined_reward = combined_reward_list[i]

            reward_extra_info["combined_reward"].append(combined_reward)
            reward_tensor[i, valid_response_length_list[i] - 1] = combined_reward

            prompt_str = prompt_str_list[i]
            response_str = response_str_list[i]
            ground_truth = ground_truth_list[i]
            extra_info = extra_info_list[i]
            data_source = data_source_list[i]
            mcp_format_reward = tool_format_reward_list[i]
            accuracy_reward = reward_extra_info["accuracy_reward"][i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[accuracy_reward]", accuracy_reward)
                print("[tool_format_reward]", mcp_format_reward)
                print("[combined_reward]", combined_reward)
                if isinstance(combined_reward, dict):
                    for key, value in combined_reward.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", combined_reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor