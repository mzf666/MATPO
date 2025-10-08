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
import os

from anthropic import Anthropic
from fastmcp import FastMCP

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Initialize FastMCP server
mcp = FastMCP("reasoning-mcp-server")


@mcp.tool()
async def reasoning(question: str) -> str:
    """You can use this tool use solve hard math problem, puzzle, riddle and IQ test question that requries a lot of chain of thought efforts.
    DO NOT use this tool for simple and obvious question.

    Args:
        question: The hard question.

    Returns:
        The answer to the question.
    """

    messages_for_llm = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question,
                }
            ],
        }
    ]

    client = Anthropic()
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=21000,
        thinking={
            "type": "enabled",
            "budget_tokens": 19000,
        },
        messages=messages_for_llm,
        stream=False,
    )

    try:
        return response.content[-1].text
    except Exception:
        # Handle the case where the text just missing
        print("Reasoning Error: only thinking content is returned")
        return response.content[-1].thinking


if __name__ == "__main__":
    mcp.run(transport="stdio")
