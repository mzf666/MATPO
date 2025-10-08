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

import asyncio
import logging
import os
import random
import re
import time
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import ray
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def is_timeout_error(error: Exception) -> bool:
    """Check if an error is a timeout-related error."""
    error_str = str(error)
    return any(keyword in error_str for keyword in ["ETIMEDOUT", "ECONNRESET", "Timeout", "Timed out"])


class MCPTool(BaseTool):
    """A tool for calling MCP tools.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Inner retry loop (for tool calls) settings
        self.max_retries = config.get("max_retries", 5)
        self.base_delay_between_retries = config.get("base_delay_between_retries", 1)  # in seconds

        # Outer retry loop (for process startup) settings
        self.startup_max_retries = config.get("startup_max_retries", 3)
        self.startup_base_delay_between_retries = config.get("startup_base_delay_between_retries", 2)  # in seconds

        self.connection_timeout = config.get("connection_timeout", 30)
        self.execution_timeout = config.get("execution_timeout", 180)

        self.is_agent_tool = self.tool_schema.type == "agent"

        # Get command, args, and env from config
        if not self.is_agent_tool:
            self.params = StdioServerParameters(
                command=config["command"],
                args=config.get("args",[]),
                env={e: os.environ.get(e) for e in config["env"]} if "env" in config.keys() else {},
            )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        # TODO: Add all create_kwargs to dict
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        if self.is_agent_tool:
            raise ValueError("Sub-agent execution unexpectly started!")

        response = ""

        # Outer retry loop for subprocess startup with exponential backoff and jitter
        for startup_attempt in range(self.startup_max_retries):
            try:
                logger.warning(
                    f"Tool startup attempt {startup_attempt + 1}/{self.startup_max_retries} for command '{self.params.command}'"
                )
                async with stdio_client(self.params) as (read, write):
                    # Startup successful, now perform the operation with its own inner retry loop
                    async with ClientSession(
                        read, write, read_timeout_seconds=timedelta(seconds=self.connection_timeout)
                    ) as session:
                        for attempt in range(self.max_retries):
                            try:
                                await session.initialize()
                                result = await session.call_tool(
                                    self.name,
                                    arguments=parameters,
                                    read_timeout_seconds=timedelta(seconds=self.execution_timeout),
                                )
                                response = result.content[0].text if result.content else ""
                                if attempt > 0:
                                    logger.warning(f"Tool call for '{self.name}' succeeded on attempt {attempt + 1}")
                                self._instance_dict[instance_id]["response"] = response
                                return response, 0.0, {}  # SUCCESS: return immediately
                            except Exception as e:
                                if is_timeout_error(e) and attempt < self.max_retries - 1:
                                    delay = self.base_delay_between_retries * (2**attempt) + random.uniform(0, 1)
                                    logger.warning(
                                        f"Tool call for '{self.name}' failed on attempt {attempt + 1}: {e}. Retrying in {delay:.2f} seconds..."
                                    )
                                    await asyncio.sleep(delay)
                                else:
                                    response = f"Tool call for '{self.name}' failed after {attempt + 1} attempts: {e}"
                                    logger.error(response)
                                    try:
                                        mcp_tool_checker = ray.get_actor(name="mcp_tool_checker", namespace="monitor")
                                        ray.get(mcp_tool_checker.record_error.remote(str(e), self.name))
                                    except Exception as e_checker:
                                        print(f"MCP tool checker not found, skip recording error: {e_checker}")
                                    self._instance_dict[instance_id]["response"] = response
                                    return response, 0.0, {}  # INNER FAIL: return immediately

            except Exception as e:
                if startup_attempt < self.startup_max_retries - 1:
                    delay = self.startup_base_delay_between_retries * (2**startup_attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Tool subprocess startup failed on attempt {startup_attempt + 1}/{self.startup_max_retries} with error: {e}. Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    response = f"Tool subprocess failed to start after {self.startup_max_retries} attempts: {e}"
                    logger.error(f"Giving up on starting tool '{self.params.command}': {e}")
                    self._instance_dict[instance_id]["response"] = response
                    return response, 0.0, {}  # OUTER FAIL: return immediately

        self._instance_dict[instance_id]["response"] = response
        return response, 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
