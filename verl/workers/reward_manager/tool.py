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

import ast
import asyncio
import io
import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List

import regex
import torch
from openai import AsyncOpenAI
from tqdm import tqdm

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.llm_judge import SIMPLEQA_EVALUATION_PROMPT, extract_boxed_answer
from verl.workers.reward_manager import register


async def batch_compute_scores_async(client, data_source_list, response_str_list, ground_truth_list, extra_info_list, batch_size=10, max_retry=3):
    """
    Compute scores asynchronously using OpenAI API in batches.
    
    Args:
        client: AsyncOpenAI client instance
        data_source_list: List of data sources
        response_str_list: List of response strings
        ground_truth_list: List of ground truth values
        extra_info_list: List of extra info
        batch_size: Number of concurrent requests
        max_retry: Maximum number of retries for failed requests
    
    Returns:
        List of accuracy rewards
    """
    async def compute_single_score(i, retry_count=0):
        """Compute score for a single sample with retry logic"""
        try:
            data_source = data_source_list[i]
            response_str = response_str_list[i]
            ground_truth = ground_truth_list[i]
            extra_info = extra_info_list[i]
            
            prompt = SIMPLEQA_EVALUATION_PROMPT.format(extra_info.get("question", ""), ground_truth, extract_boxed_answer(response_str))

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2
            )

            content = response.choices[0].message.content
            match = re.search(r"(A|B|C)", content)

            if match:
                grade_map = {"A": 1.0, "B": 0.0, "C": 0.0}
                return (i, grade_map[match.group(0)])
            else:
                return (i, 0.0)
                
        except Exception as e:
            print(f"Error computing score for sample {i} (attempt {retry_count + 1}): {e}")
            
            # Check if it's a rate limit error
            error_str = str(e).lower()
            is_rate_limit = any(keyword in error_str for keyword in ['rate', 'limit', 'quota', '429', 'too many requests'])
            is_connection_error = any(keyword in error_str for keyword in ['connection', 'timeout', 'network'])
            
            # Retry if we haven't reached max_retry
            if retry_count < max_retry:
                print(f"Retrying sample {i} (attempt {retry_count + 2}/{max_retry + 1})")
                
                # Use longer wait time for rate limit errors
                if is_rate_limit:
                    wait_time = 30 * (2 ** retry_count)  # 30s, 60s, 120s for rate limits
                else:
                    wait_time = 10 * (2 ** retry_count)  # 10s, 20s, 40s for other errors
                
                await asyncio.sleep(wait_time)
                
                return await compute_single_score(i, retry_count + 1)
            else:
                print(f"Max retry reached for sample {i}")
                raise Exception(f"Failed to compute score for sample {i} after {max_retry + 1} attempts")
    
    # Process in batches to avoid overwhelming the API
    results_dict = {}
    for i in range(0, len(data_source_list), batch_size):
        batch_indices = list(range(i, min(i + batch_size, len(data_source_list))))
        
        # Create tasks for the current batch
        tasks = [compute_single_score(idx) for idx in batch_indices]
        
        # Execute batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Add small delay between batches to prevent rate limiting
        if i + batch_size < len(data_source_list):
            await asyncio.sleep(3)  # Increased from 1s to 3s
        
        # Handle any exceptions in the batch and store results by index
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                print(f"Exception in batch {i//batch_size}, sample {batch_indices[j]}: {result}")
                # Re-raise the exception instead of using fallback
                raise result
            else:
                idx, score = result
                results_dict[idx] = score
    
    # Reconstruct results list in the correct order
    results = [results_dict[i] for i in range(len(data_source_list))]
    
    # Validate that results match the input data length
    if len(results) != len(response_str_list):
        raise ValueError(f"Results length mismatch: got {len(results)}, expected {len(response_str_list)}")
    
    return results


def create_async_openai_client(api_key=None):
    """
    Create an AsyncOpenAI client instance.
    
    Args:
        api_key: OpenAI API key. If None, will try to get from environment.
    
    Returns:
        AsyncOpenAI client instance
    """
    return AsyncOpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY")
    )


async def close_openai_client(client):
    """
    Properly close OpenAI client connections.
    """
    try:
        if hasattr(client, '_client') and hasattr(client._client, 'close'):
            await client._client.close()
        elif hasattr(client, 'close'):
            await client.close()
    except Exception as e:
        print(f"Warning: Error closing OpenAI client: {e}")


def run_async_computation(client, data_source_list, response_str_list, ground_truth_list, extra_info_list, batch_size=10, max_retry=3):
    """
    Run async computation in a new event loop.
    
    Args:
        client: AsyncOpenAI client instance
        data_source_list: List of data sources
        response_str_list: List of response strings
        ground_truth_list: List of ground truth values
        extra_info_list: List of extra info
        batch_size: Number of concurrent requests
        max_retry: Maximum number of retries for failed requests
    
    Returns:
        List of accuracy rewards
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Run the batch computation
        results = loop.run_until_complete(
            batch_compute_scores_async(
                client,
                data_source_list,
                response_str_list,
                ground_truth_list,
                extra_info_list,
                batch_size,
                max_retry
            )
        )
        
        # Close the client connections properly
        loop.run_until_complete(close_openai_client(client))
        
        return results
    finally:
        loop.close()




@register("tool")
class ToolRewardManager:
    """The reward manager for tool data."""

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
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.accuracy_reward_weight = accuracy_reward_weight  # Store the weight for the accuracy reward
        self.tool_format_reward_weight = tool_format_reward_weight  # Store the weight for the tool reward
        self.gate_tool_format_reward = gate_tool_format_reward  # Store the gating flag
        self.async_process = async_process  # whether use async process to compute reward
        self.batch_size = batch_size  # batch size for async processing
        self.max_retry = max_retry  # max retry for async processing
        
        # Store API key for creating fresh clients
        if self.async_process:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def __call__(self, data: DataProto, return_dict=False):
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

            ## TODO: check if the tool-format is correctly implemented for browsing-agent
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

    
    def check_mcp_format(self, prompt_str, response_str, verbose=False):

        """
        Returns a binary: 
            1.0: conversation follows mcp format 
            0.0: conversation does not follow mcp format
        """ 

        def parse_tool_schemas_with_types(system_content: str) -> Dict[str, dict]:
            tool_specs = {}

            server_tool_blocks = re.findall(
                r"## Server name: (.+?)\n((?:.|\n)*?)(?=## Server name:|\Z)",
                system_content
            )

            for server_name, tool_section in server_tool_blocks:
                tool_defs = re.findall(
                    r"### Tool name: (.*?)\n.*?Input JSON schema: (.*?)(?=\n### Tool name:|\n## Server name:|\Z)",
                    tool_section,
                    re.DOTALL
                )

                for tool_name, raw_schema in tool_defs:
                    tool_name = tool_name.strip()
                    server_name = server_name.strip()
                    raw_schema = raw_schema.strip()

                    try:
                        schema_dict = ast.literal_eval(raw_schema)
                        required_fields = schema_dict.get("required", [])
                        properties = schema_dict.get("properties", {})

                        prop_types = {
                            k: v.get("type", "unknown")
                            for k, v in properties.items()
                            if isinstance(v, dict)
                        }

                    except Exception:
                        # Fallback: extract 'required'
                        match = re.search(r"'required'\s*:\s*\[(.*?)\]", raw_schema)
                        if match:
                            items = match.group(1)
                            required_fields = [x.strip().strip("'\"") for x in items.split(',') if x.strip()]
                        else:
                            required_fields = f"FAILED TO PARSE: fallback failed"

                        # Fallback: extract property types manually
                        prop_types = {}
                        prop_block = re.search(r"'properties'\s*:\s*{(.*?)}\s*,?\s*'required'", raw_schema, re.DOTALL)
                        if prop_block:
                            props_text = prop_block.group(1)
                            prop_defs = re.findall(
                                r"'(\w+)'\s*:\s*{[^}]*?'type'\s*:\s*'(\w+)'", props_text
                            )
                            prop_types = {k: v for k, v in prop_defs}
                        else:
                            prop_types = "FAILED TO PARSE properties"

                    tool_specs[tool_name] = {
                        "server_name": server_name,
                        "required": required_fields,
                        "properties": prop_types
                    }

            return tool_specs

        def extract_assistant_msg(msg: str) -> List[str]:
            msg = msg.strip()

            if not msg.startswith("\nassistant\n"):
                msg = "\nassistant\n" + msg

            pattern = r'\nassistant\n.*?(?=\nuser\n|$)'
            assistant_blocks = re.findall(pattern, msg, re.DOTALL)

            return "\n".join(assistant_blocks)

        def extract_tool_blocks(msg: str) -> List[str]:
            return re.findall(r"<use_mcp_tool>.*?</use_mcp_tool>", msg.strip(), re.DOTALL)


        def validate_mcp_tool_use(xml_str: str, tool_schemas: dict) -> dict:
            """
            Validates a <use_mcp_tool>...</use_mcp_tool> string against the given tool_schemas.
            Checks:
            - tool and server match
            - required fields exist
            - argument types match
            Returns dict with `valid: bool` and `errors: List[str]`
            """
            errors = []

            # Extract server_name, tool_name, and arguments block
            try:
                server_name = re.search(r"<server_name>(.*?)</server_name>", xml_str).group(1).strip()
                tool_name = re.search(r"<tool_name>(.*?)</tool_name>", xml_str).group(1).strip()
                arguments_raw = re.search(r"<arguments>\s*(\{.*\})\s*</arguments>", xml_str, re.DOTALL).group(1)
                arguments = json.loads(arguments_raw)
            except Exception as e:
                return {"valid": False, "errors": [f"Failed to parse XML or JSON: {e}"]}
            
            errors.append(f'{tool_name}')

            # Validate tool existence
            if tool_name not in tool_schemas:
                return {"valid": False, "errors": [f"Tool '{tool_name}' not found in schema."]}

            schema = tool_schemas[tool_name]

            # Validate server
            if schema["server_name"] != server_name:
                errors.append(f"Server name mismatch: expected '{schema['server_name']}', got '{server_name}'")

            # Validate required fields
            required = schema.get("required", [])
            for field in required:
                if field not in arguments:
                    errors.append(f"Missing required field: '{field}'")

            # Validate argument types
            type_mapping = {
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
                "object": dict,
                "array": list,
                "null": type(None),
            }

            properties = schema.get("properties", {})
            if isinstance(properties, dict):
                for k, v in arguments.items():
                    expected_type = properties.get(k)
                    if expected_type:
                        if not isinstance(v, type_mapping.get(expected_type, object)):
                            errors.append(f"Field '{k}' should be of type '{expected_type}', got '{type(v).__name__}'")
                    else:
                        errors.append(f"Unexpected argument: '{k}'")

            return {"valid": len(errors) == 1, "errors": errors}

        # Extract tools given to the agent, keeping the required mcp format details
        tool_specs = parse_tool_schemas_with_types(prompt_str) 
        if not tool_specs:
            raise ValueError("No tool schemas found. This may not be a valid system prompt.")

        # Extract assistant message
        assistant_str = extract_assistant_msg(response_str)

        # Check validity of mcp format
        blocks = extract_tool_blocks(assistant_str)
        if len(blocks) == 0:
            format_score = 1.0
            return format_score

        format_score = 0.0
        for block in blocks:
            # Validate right format
            check = validate_mcp_tool_use(block, tool_specs)
            if not check['valid']:
                if verbose:
                    print(f"{check['errors'][0]}: {check['errors'][1:]}")
                format_score += 0.0
            else:
                format_score += 1.0

        format_score /= len(blocks)
                
        return format_score