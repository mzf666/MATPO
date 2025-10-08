# Copyright 2025 MiroMind Team
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
import json
import time
from verl.workers.reward_manager.tool import run_async_computation, gaia_naive_compute_score
from verl.utils.reward_score.llm_judge import extract_boxed_answer
from openai import AsyncOpenAI
import os
from unittest.mock import Mock, patch

def build_data(num_test_samples):
    # read file from testData.jsonl
    data = []
    with open("/pfs/training-data/xingxuanli/GitHub/verl-tool/verl/workers/reward_manager/tests/testData.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))

    prompt_str_list = [item["prompt_str"] for item in data[:num_test_samples]]
    response_str_list = [item["response_str"] for item in data[:num_test_samples]]
    ground_truth_list = [item["ground_truth"] for item in data[:num_test_samples]]
    data_source_list = [item["data_source"] for item in data[:num_test_samples]]
    extra_info_list = [item["extra_info"] for item in data[:num_test_samples]]
    valid_response_length_list = [item["valid_response_length"] for item in data[:num_test_samples]]

    return prompt_str_list, response_str_list, ground_truth_list, data_source_list, extra_info_list, valid_response_length_list
    

# test the async process function
def async_process_function(prompt_str_list, response_str_list, ground_truth_list, data_source_list, extra_info_list, valid_response_length_list, openai_client, batch_size=100, max_retry=3):
    start_time = time.time()
    print("Start computing accuracy reward with async api, this may take a while...")
    print(f"Async configuration: batch_size={batch_size}, max_retry={max_retry}")
    try:
        # Run async computation using the new async functions
        accuracy_reward_list = run_async_computation(
            openai_client,
            data_source_list,
            response_str_list,
            ground_truth_list,
            extra_info_list,
            batch_size,
            max_retry
        )
        
        end_time = time.time()
        print(f"Async batch processing completed in {end_time - start_time:.2f} seconds")

    except Exception as e:
        # if error, we use gaia_naive_compute_score to compute accuracy reward
        print(f"Error in batch api: {e}")
        print("Falling back to gaia_naive_compute_score...")
        accuracy_reward_list = []
        for i in range(len(response_str_list)):
            accuracy_reward = gaia_naive_compute_score(
                data_source=data_source_list[i],
                solution_str=response_str_list[i],
                ground_truth=ground_truth_list[i],
                extra_info=extra_info_list[i],
            )
            accuracy_reward_list.append(accuracy_reward)

    for i in range(len(accuracy_reward_list)):
        extracted = extract_boxed_answer(response_str_list[i])
        ground_truth = ground_truth_list[i]
        reward = accuracy_reward_list[i]
        print(f"{extracted:<40} | {ground_truth:<40} | {reward}")


def test_exception_handling(num_test_samples):
    """Test the exception handling and fallback mechanism"""
    print("=== Testing Exception Handling ===")
    
    # Load test data
    prompt_str_list, response_str_list, ground_truth_list, data_source_list, extra_info_list, valid_response_length_list = build_data(num_test_samples)
    
    # Create a mock OpenAI client that always raises an exception
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("API Error: Rate limit exceeded")
    
    # Call the function with the mock client that will fail
    async_process_function(
        prompt_str_list, 
        response_str_list, 
        ground_truth_list, 
        data_source_list, 
        extra_info_list, 
        valid_response_length_list, 
        mock_client, 
        batch_size=100, 
        max_retry=1
    )

    print("=== Exception handling test completed ===")




if __name__ == "__main__":
    num_test_samples = 128 # max is 128
    test_mode = "exception" # "normal" or "exception"

    if test_mode == "normal":
        print("=== Normal Operation Test ===")
        prompt_str_list, response_str_list, ground_truth_list, data_source_list, extra_info_list, valid_response_length_list = build_data(num_test_samples)

        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        async_process_function(prompt_str_list, response_str_list, ground_truth_list, data_source_list, extra_info_list, valid_response_length_list, openai_client, batch_size=100, max_retry=3)
        print("=== Normal Operation Test Completed ===")
    
    elif test_mode == "exception":
        test_exception_handling(num_test_samples)

    else:
        raise ValueError(f"Invalid test mode: {test_mode}")