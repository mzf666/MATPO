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
import difflib
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, model_validator
from transformers import PreTrainedTokenizer

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BASE_CHAT_HISTORY = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "I am a user."}]


class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""

    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")


class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""

    batch_data_id: int = 0
    batch_data_uid: str = None # for identifying the rollout <--> reqs_from_subagents relationship
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    tool_schemas: Optional[List[OpenAIFunctionToolSchema]] = None
    tools_kwargs: Dict[str, Any] = {}
    input_ids: List[int]
    prompt_ids: List[int]
    response_ids: List[int]
    attention_mask: List[int]
    prompt_attention_mask: List[int]
    response_attention_mask: List[int]
    position_ids: List[int]
    prompt_position_ids: List[int]
    response_position_ids: List[int]
    loss_mask: List[int]
    prompt_loss_mask: List[int]
    response_loss_mask: List[int]
    reward_scores: Dict[str, float]
    max_prompt_len: int
    max_response_len: int = 8192
    max_model_len: int = 32768
    metrics: Dict[str, List[Any]] = {}
    reqs_from_subagents: Optional[List[Dict[str, Any]]] = None  # For tracking agent call hierarchy, to edit
    is_from_subagent_tool: bool = False
    parent_req_id: str = None
    turn_count: int = None # for tracking the turn count of the request from the subagent tool call raised by the main agent

    use_inference_chat_template: bool
    enable_tokenization_sanity_check: bool
    generation_prompt_ids: List[int]
    base_conv_wo_gen_prompt_end_pos: int
    base_conv_with_gen_prompt_end_pos: int

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        if not (messages := values.get("messages")):
            raise ValueError("messages is required for AsyncRolloutRequest initialization")
        if not (max_prompt_len := values.get("max_prompt_len")):
            raise ValueError("max_prompt_len is required for AsyncRolloutRequest initialization")
        if not (tokenizer := values.pop("tokenizer", None)):
            raise ValueError("tokenizer is required for AsyncRolloutRequest initialization")

        values["messages"] = [Message.model_validate(msg) for msg in messages]

        use_mcp_tool_call = values.get("use_mcp_tool_call", False)

        tools = [tool.model_dump() for tool in tool_schemas] if ((tool_schemas := values.get("tool_schemas", [])) and (not use_mcp_tool_call)) else None
        tokens_without_prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=True)
        if not values.get("input_ids") or not values.get("attention_mask"):
            tokenization_dict_with_prompt = tokenizer.apply_chat_template(messages, tools=([tool.model_dump() for tool in tool_schemas] if (not use_mcp_tool_call) else None), add_generation_prompt=True, tokenize=True, return_dict=True)
            values["input_ids"], values["attention_mask"] = tokenization_dict_with_prompt["input_ids"], tokenization_dict_with_prompt["attention_mask"]
            if len(values["input_ids"]) > max_prompt_len:
                # Only log the warning to avoid truncating in the middle of generation prompt. Consider raising an error for this case in the future.
                logger.warning(f"Prompt {values['batch_data_id']} length {len(values['input_ids'])} greater than max_prompt_len {max_prompt_len} after applied chat template with tools.")

        values["prompt_ids"], values["prompt_attention_mask"] = values["input_ids"], values["attention_mask"]
        values["position_ids"] = values["prompt_position_ids"] = compute_position_id_with_mask(torch.tensor(values["attention_mask"])).tolist()
        values["loss_mask"] = values["prompt_loss_mask"] = [0] * len(values["input_ids"])
        values["generation_prompt_ids"] = values["input_ids"][len(tokens_without_prompt) :]
        values["base_conv_wo_gen_prompt_end_pos"] = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=False, tokenize=False))
        values["base_conv_with_gen_prompt_end_pos"] = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=True, tokenize=False))
        return values

    def _update_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request in additive manner.
        """
        self.input_ids += new_input_ids
        attention_mask = [int(attention_mask)] * len(new_input_ids)
        self.attention_mask += attention_mask
        self.loss_mask += [int(loss_mask)] * len(new_input_ids)
        self.position_ids += (compute_position_id_with_mask(torch.tensor(attention_mask)) + (self.position_ids[-1] + 1)).tolist()

        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def get_generation_prompt_ids(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        generation_prompt_ids = [] if self.input_ids[-len(self.generation_prompt_ids) :] == self.generation_prompt_ids else self.generation_prompt_ids
        if generation_prompt_ids:
            self._update_input_ids(generation_prompt_ids, attention_mask=True, loss_mask=False)

        if self.use_inference_chat_template:
            return tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=True, tokenize=True)
        else:
            return self.input_ids

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
        use_mcp_tool_call: bool = False,
        keep_think_text_for_last_round_only: bool = False,
        think_block_close_tag: str = "</think>",
    ) -> None:
        if keep_think_text_for_last_round_only and tool_calls is not None:  # assistant message w/ tool calls is not the last round
            content = content.split(think_block_close_tag)[-1].lstrip("\n")  # use the same strip logic of Qwen3 chat template

        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))
        if use_mcp_tool_call:
            last_message = Message(role="assistant", content=content)
        else:
            last_message = self.messages[-1]
        content = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, last_message], tools=([tool.model_dump() for tool in self.tool_schemas] if (not use_mcp_tool_call and self.tool_schemas) else None), add_generation_prompt=False, tokenize=False)
        content_ids = tokenizer.encode(content[self.base_conv_with_gen_prompt_end_pos :], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=True)

    def add_tool_response_messages(self, tokenizer: PreTrainedTokenizer, contents: list[str], use_mcp_tool_call: bool = False) -> None:
        if not contents:
            return
        if use_mcp_tool_call:
            self.messages.extend([Message(role="user", content=content) for content in contents])
        else:
            self.messages.extend([Message(role="tool", content=content) for content in contents])
        content = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, *self.messages[-len(contents) :]], tools=([tool.model_dump() for tool in self.tool_schemas] if (not use_mcp_tool_call and self.tool_schemas) else None), add_generation_prompt=False, tokenize=False)
        content_ids = tokenizer.encode(content[self.base_conv_wo_gen_prompt_end_pos :], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """
        metrics: should be a dict of tools_name -> Any
        """
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)

    def tokenization_sanity_check(self, tokenizer: PreTrainedTokenizer, use_mcp_tool_call: bool = False, ignore_think_block: bool = True):
        # Generate tokens based on current messages
        full_tokens = tokenizer.apply_chat_template(
            [msg.model_dump() for msg in self.messages],
            tools=([tool.model_dump() for tool in self.tool_schemas] if (not use_mcp_tool_call and self.tool_schemas) else None),
            add_generation_prompt=False,
            tokenize=True
        )
        
        # Early return if sequences match
        if self.input_ids == full_tokens:
            return
            
        # Pattern to identify thinking blocks
        pattern = r'<think>.+?</think>\n\n'
        s = difflib.SequenceMatcher(None, self.input_ids, full_tokens)
        
        # Iterate over the opcodes generated by difflib.SequenceMatcher to check for differences
        # between input_ids and full_tokens.
        # Each opcode is a tuple: (tag, i1, i2, j1, j2)
        #   - tag: the type of difference ('replace', 'delete', 'insert', 'equal')
        #   - i1, i2: the start and end indices in input_ids for this operation
        #   - j1, j2: the start and end indices in full_tokens for this operation
        # This allows us to precisely locate and analyze mismatches between the two token sequences,
        # and log detailed information for debugging tokenization inconsistencies.
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == 'equal':
                continue  # skip equal parts
            elif tag == 'delete':
                diff_text = tokenizer.decode(self.input_ids[i1:i2], skip_special_tokens=True)
                if re.fullmatch(pattern, diff_text, re.DOTALL):
                    if not ignore_think_block:
                        logger.warning(f"Inconsistent tokenization: input_ids contain extra <think>...</think> block not in full_tokens")
                else:
                    logger.warning(f"Inconsistent tokenization: input_ids contain extra text: `{diff_text}`")
            elif tag == 'insert':
                diff_text = tokenizer.decode(full_tokens[j1:j2], skip_special_tokens=True)
                logger.warning(f"Inconsistent tokenization: full_tokens contain extra text: `{diff_text}`")
            elif tag == 'replace':
                diff_text_input_ids = tokenizer.decode(self.input_ids[i1:i2], skip_special_tokens=True)
                diff_text_full_tokens = tokenizer.decode(full_tokens[j1:j2], skip_special_tokens=True)
                logger.warning(f"Inconsistent tokenization: `{diff_text_input_ids}` in input_ids replaced by `{diff_text_full_tokens}` in full_tokens")

    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, float],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
        use_mcp_tool_call: bool = False,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        if self.enable_tokenization_sanity_check:
            self.tokenization_sanity_check(tokenizer, use_mcp_tool_call)
            
        # In case we failed to generate the assistant message and the generation prompt ids were already added to input_ids, remove them from the end of input_ids
        if self.input_ids[-len(self.generation_prompt_ids) :] == self.generation_prompt_ids:
            self.input_ids = self.input_ids[: -len(self.generation_prompt_ids)]
            self.attention_mask = self.attention_mask[: -len(self.generation_prompt_ids)]
            self.position_ids = self.position_ids[: -len(self.generation_prompt_ids)]
            self.loss_mask = self.loss_mask[: -len(self.generation_prompt_ids)]

        self.response_ids = self.input_ids[len(self.prompt_ids) :]
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
        self.truncate_output_ids(tokenizer)
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def truncate_messages(self, tokenizer: PreTrainedTokenizer, max_input_len: int, use_mcp_tool_call: bool, keep_think_text_for_last_round_only: bool, think_block_close_tag: str):
        """
        Truncates messages to fit within max_input_len by replacing the content of old turns
        with a placeholder message. This preserves the conversation structure.
        """
        # 1. Check if truncation is needed.
        current_token_len = len(self.get_generation_prompt_ids(tokenizer))
        # tokenizer.decode(self.get_generation_prompt_ids(tokenizer))
        if current_token_len <= max_input_len:
            return

        # 2. Perform message list truncation turn by turn from the start.
        messages_to_truncate = self.messages
        final_messages = self.messages
        if len(messages_to_truncate) > 3:
            preserved_start = messages_to_truncate[:2]
            preserved_end = messages_to_truncate[-1]
            middle_messages = messages_to_truncate[2:-1]

            while current_token_len > max_input_len:
                if not middle_messages:
                    break
                middle_messages.pop()

                final_messages = preserved_start + middle_messages + [preserved_end]
                self.rebuild_messages(final_messages, tokenizer, use_mcp_tool_call, keep_think_text_for_last_round_only, think_block_close_tag)
                input_ids_tmp = self.get_generation_prompt_ids(tokenizer)
                current_token_len = len(input_ids_tmp)
                # tokenizer.decode(input_ids_tmp)
                
    def rebuild_messages(self, new_messages: List[Message], tokenizer: PreTrainedTokenizer, use_mcp_tool_call: bool, keep_think_text_for_last_round_only: bool, think_block_close_tag: str):
        # Re-initialize state and replay the history.
        retained_messages = [msg.model_copy(deep=True) for msg in new_messages]
        initial_messages = retained_messages[:2]
        self.messages = initial_messages

        # Re-run the core logic from the @model_validator for the initial two messages
        # The same logic as the initialize_request() method
        tools = [tool.model_dump() for tool in self.tool_schemas] if (self.tool_schemas and not use_mcp_tool_call) else None
        tokens_without_prompt = tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=tools, add_generation_prompt=False, tokenize=True)
        tokenization_dict = tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=tools, add_generation_prompt=True, tokenize=True, return_dict=True)

        self.input_ids = tokenization_dict["input_ids"]
        self.attention_mask = tokenization_dict["attention_mask"]
        self.prompt_ids = list(self.input_ids)
        self.prompt_attention_mask = list(self.attention_mask)
        self.position_ids = compute_position_id_with_mask(torch.tensor(self.attention_mask)).tolist()
        self.prompt_position_ids = list(self.position_ids)
        self.loss_mask = [0] * len(self.input_ids)
        self.prompt_loss_mask = list(self.loss_mask)
        self.generation_prompt_ids = self.input_ids[len(tokens_without_prompt):]
        
        # 4. Replay the rest of the messages to correctly rebuild the state using existing APIs
        messages_to_replay = retained_messages[2:]
        i = 0
        while i < len(messages_to_replay):
            message = messages_to_replay[i]
            
            if message.role == 'assistant':
                self.add_assistant_message(
                    tokenizer, 
                    message.content, 
                    message.tool_calls, 
                    use_mcp_tool_call=use_mcp_tool_call,
                    keep_think_text_for_last_round_only=keep_think_text_for_last_round_only,
                    think_block_close_tag=think_block_close_tag
                )
                i += 1
            elif message.role == 'tool' or (use_mcp_tool_call and message.role == 'user'):
                contents_to_add = []
                current_role = message.role
                
                while i < len(messages_to_replay) and messages_to_replay[i].role == current_role:
                    contents_to_add.append(messages_to_replay[i].content)
                    i += 1
                
                self.add_tool_response_messages(
                    tokenizer,
                    contents_to_add,
                    use_mcp_tool_call=use_mcp_tool_call,
                )
            else:
                logger.warning(f"Unexpected message role '{message.role}' during message truncation replay.")
                i += 1

    def truncate_output_ids(self, tokenizer: PreTrainedTokenizer) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
