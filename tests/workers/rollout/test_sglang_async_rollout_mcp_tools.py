# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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
# Adapted from tests/workers/rollout/test_sglang_async_rollout_sf_tools.py


from unittest.mock import patch

import numpy as np
import pytest
from tensordict import TensorDict
from transformers import AutoConfig, AutoTokenizer
from utils_sglang import (
    get_rollout_config,
    prepare_inputs,
)

from verl.protocol import DataProto
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)
from verl.workers.rollout.schemas import AsyncRolloutRequestStateEnum
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)
user_content = DEFAULT_USER_CONTENT_PREFIX.rstrip("\n") + "How's the weather lately?"


def get_search_messages():
    user_prompt = {
        "role": "user",
        "content": user_content,
    }

    expect_turn_0_msg = {
        "role": "assistant",
        "content": "Let me search the web.",
        "tool_calls": [{"type": "function", "function": {"name": "google_search", "arguments": {"q": "today's weather", "gl": "us", "hl": "en"}}}],
    }

    expect_turn_1_msg = {
        "role": "assistant",
        "content": "Let me search again.",
        "tool_calls": [{"type": "function", "function": {"name": "google_search", "arguments": {"q": "tomorrow's weather", "gl": "us", "hl": "en"}}}],
    }

    expect_turn_2_msg = {
        "role": "assistant",
        "content": "<answer>Today is sunny and tomorrow will be cloudy in Beijing.</answer>",
    }

    # Mock search tool responses
    tool_return_0_msg = {"role": "tool", "content": "Today's weather in Beijing is sunny."}
    tool_return_1_msg = {"role": "tool", "content": "Tomorrow's weather in Beijing is cloudy."}

    user_prompts = [user_prompt]
    expect_turn_array = [expect_turn_0_msg, expect_turn_1_msg, expect_turn_2_msg]
    tool_return_array = [tool_return_0_msg, tool_return_1_msg]

    return user_prompts, expect_turn_array, tool_return_array


class TestRolloutWithMCPTools:
    @pytest.fixture
    def qwen_tokenizer(self):
        local_model_path = "Qwen/Qwen2.5-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    # we only need this for tokenizer
    @pytest.fixture
    def qwen_model_config(self):
        local_model_path = "Qwen/Qwen2.5-0.5B"
        config = AutoConfig.from_pretrained(local_model_path)
        return config

    @pytest.fixture
    def search_data(self, qwen_tokenizer):
        user_prompt, expect_turn_array, tool_return_array = get_search_messages()
        prompts = [[message] for message in user_prompt]
        preencode_turn_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=False) for turn in expect_turn_array]
        preencode_tool_return_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=True) for turn in tool_return_array]
        return prompts, preencode_turn_array, preencode_tool_return_array

    @pytest.fixture
    def search_rollout_config(self):
        max_prompt_length = 4096
        max_response_length = 3000
        dtype = "bfloat16"
        tensor_parallel_size = 1
        tool_path = "./resource/tool_configs/mcp_tool_config"
        rollout_config = get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size, tool_path)
        return rollout_config

    @pytest.fixture
    def search_data_proto(self, search_data, qwen_tokenizer):
        preencode_prompts, _, _ = search_data
        prompts = [qwen_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in preencode_prompts]
        input_ids, attention_mask, position_ids = prepare_inputs(qwen_tokenizer, prompts, 1000)
        prompt_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        messages = np.asarray(preencode_prompts)

        tools_kwargs = np.array(
            [
                {
                    "google_search": {
                        "create_kwargs": {"ground_truth": "Today is sunny and tomorrow will be cloudy in Beijing.", "data_source": "searchR1_nq"},
                    },
                    "scrape": {
                        "create_kwargs": {"ground_truth": "Today is sunny and tomorrow will be cloudy in Beijing.", "data_source": "searchR1_nq"},
                    },
                }
            ],
            dtype=object,
        )
        index = np.array([0], dtype=object)
        prompts = DataProto(batch=prompt_dict, non_tensor_batch={"raw_prompt": messages, "tools_kwargs": tools_kwargs, "index": index})
        return prompts

    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_tools_registration(self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config):
        rollout = SGLangRollout(actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        assert len(rollout._tool_schemas) == 2
        assert "google_search" in rollout._tool_map.keys() and "scrape" in rollout._tool_map.keys()
        from verl.tools.mcp_tool import MCPTool

        assert isinstance(rollout._tool_map["google_search"], MCPTool)
        assert isinstance(rollout._tool_map["scrape"], MCPTool)
        # depend on the tokenizer
        assert rollout._tool_call_parser_type == "qwen25"

    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_rollout_req_creation(self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto):
        rollout = SGLangRollout(actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        req_list = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)
        assert len(req_list) == 1
        assert req_list[0].state == AsyncRolloutRequestStateEnum.PENDING
        assert len(req_list[0].tool_schemas) == 2
        print(type(req_list[0].tool_schemas[0]), type(req_list[0].tool_schemas[1]))
        assert req_list[0].tool_schemas[0] == OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="google_search",
                description=(
                    "Tool to perform web searches via Serper API and retrieve rich results. "
                    "It is able to retrieve organic search results, people also ask, "
                    "related searches, and knowledge graph."
                ),
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "q": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Search query string"
                        ),
                        "gl": OpenAIFunctionPropertySchema(
                            type="string",
                            description=(
                                "Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')"
                            )
                        ),
                        "hl": OpenAIFunctionPropertySchema(
                            type="string",
                            description=(
                                "Optional language code for search results in ISO 639-1 format (e.g., 'en')"
                            )
                        ),
                        "location": OpenAIFunctionPropertySchema(
                            type="string",
                            description=(
                                "Optional location for search results (e.g., 'SoHo, New York, United States', "
                                "'California, United States')"
                            )
                        ),
                        "num": OpenAIFunctionPropertySchema(
                            type="number",
                            description="Number of results to return (default: 10)"
                        ),
                        "tbs": OpenAIFunctionPropertySchema(
                            type="string",
                            description=(
                                "Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, "
                                "'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year)"
                            )
                        ),
                        "page": OpenAIFunctionPropertySchema(
                            type="number",
                            description="Page number of results to return (default: 1)"
                        ),
                        "autocorrect": OpenAIFunctionPropertySchema(
                            type="boolean",
                            description="Whether to autocorrect spelling in query"
                        )
                    },
                    required=["q", "gl", "hl"]
                ),
                strict=False
            )
        )
        assert req_list[0].tool_schemas[1] == OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="scrape",
                description=(
                    "Tool to scrape a webpage and retrieve the text and, optionally, the markdown content. "
                    "It will retrieve also the JSON-LD metadata and the head metadata."
                ),
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "url": OpenAIFunctionPropertySchema(
                            type="string",
                            description="The URL of the webpage to scrape."
                        ),
                        "includeMarkdown": OpenAIFunctionPropertySchema(
                            type="boolean",
                            description="Whether to include markdown content."
                        )
                    },
                    required=["url"]
                ),
                strict=False
            )
        )
