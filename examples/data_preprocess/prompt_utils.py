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

import asyncio
import os
import time
from typing import Any, Dict, List

from omegaconf import OmegaConf


# copied from sglang_rollout.py
async def get_mcp_tools_schema(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get tool schemas from MCP server.

    Args:
        config: MCP tool configuration containing command, args, and env

    Returns:
        List of tool schemas with name, description, and parameters
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=config["command"],
        args=config.get("args", []),
        env={e: os.environ.get(e) for e in config["env"]} if "env" in config else {},
    )
    max_retries = config.get("max_retries", 5)
    delay_between_retries = config.get("delay_between_retries", 1)
    tools_schema = []

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            for attempt in range(max_retries):
                try:
                    await session.initialize()
                    response = await session.list_tools()
                    for tool in response.tools:
                        tools_schema.append(
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                                "server_name": config["server_name"],
                            }
                        )
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay_between_retries)
                    else:
                        raise
    return tools_schema


# modified based on ./examples/data_preprocess/simpleqa_multiturn_w_tool_all_mcp_ba.py
async def load_all_tool_schemas_and_kwargs(
    config: [str, OmegaConf], is_main_agent: bool
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate tool schemas and kwargs from MCP tool configuration.

    Args:
        config_path: Path to the tool configuration file
        is_main_agent: Whether the system prompt is for the main agent

    Returns:
        Tuple of (tool_schemas, tools_kwargs)
    """
    if isinstance(config, str):
        tools_config = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    else:
        tools_config = OmegaConf.to_container(config, resolve=True)

    all_tool_schemas = []
    all_tools_kwargs = {}

    # TODO: add agent-tool type key-words: different agent-tools use different set of mcp / function tools.
    
    for tool_config in tools_config["tools"]:
        tool_schema_dict = tool_config["tool_schema"]

        if tool_schema_dict["type"] == "mcp":
            # Get schemas from MCP server
            mcp_schemas = await get_mcp_tools_schema(tool_config["config"])
            all_tool_schemas.extend(mcp_schemas)

            # Add kwargs for each MCP tool
            for schema in mcp_schemas:
                all_tools_kwargs[schema["name"]] = {
                    "create_kwargs": {"ground_truth": None},  # Will be set per example
                }

        elif tool_schema_dict["type"] == "function":
            # Handle regular function tools
            function_schema = tool_schema_dict["function"]
            all_tool_schemas.append(function_schema)
            all_tools_kwargs[function_schema["name"]] = {
                "create_kwargs": {"ground_truth": None},  # Will be set per example
            }

        elif tool_schema_dict["type"] == "agent" and is_main_agent:
            # Get agent schema from config
            # NOTE: avoid agent tool call inside agent tool call
            agent_schema = tool_schema_dict["schema"]
            all_tool_schemas.append(agent_schema)

            # Add kwargs for each agent tool
            all_tools_kwargs[agent_schema["name"]] = {
                "create_kwargs": {"ground_truth": None},  # Will be set per example
            }

    agent_tools_map = {agent_config["agent_type"]: agent_config["tools"] for agent_config in tools_config.get("agents", [])}

    return all_tool_schemas, all_tools_kwargs, agent_tools_map


def format_tool_schemas_for_prompt(tool_schemas: List[Dict[str, Any]], tools_filter: List[str] = None) -> str:
    """Format tool schemas into a string for inclusion in the prompt.

    Args:
        tool_schemas: List of tool schemas

    Returns:
        Formatted string containing all tool schemas
    """
    if not tool_schemas:
        return ""

    schema_text = "\nHere are the functions available in JSONSchema format:\n\n"

    # For MCP tools, we don't have server_name in the schema, so we just list them
    last_name = None
    for schema in tool_schemas:
        if tools_filter and schema["server_name"] not in tools_filter:
            continue
        if schema["server_name"] != last_name:
            schema_text += f"## Server name: {schema['server_name']}\n"
            last_name = schema["server_name"]
        schema_text += f"### Tool name: {schema['name']}\n"
        schema_text += f"Description: {schema['description']}\n"
        schema_text += f"Input JSON schema: {schema['parameters']}\n\n"

    return schema_text


def get_agentic_system_prompt(agent_type: str = "main_agent") -> str:
    if agent_type == "main_agent":
        return """
# Agent Specific Objective

You are a task-solving agent that uses tools step-by-step to answer the user's question. Your goal is to provide complete, accurate and well-reasoned answers using additional tools.
"""
    elif agent_type == "browsing_agent":
        return """
# Agent Specific Objective

You are an agent that performs the task of searching and browsing the web for specific information and generating the desired answer. Your task is to retrieve reliable, factual, and verifiable information that fills in knowledge gaps.
Do not infer, speculate, summarize broadly, or attempt to fill in missing parts yourself. Only return factual content.

Critically assess the reliability of all information:
- If the credibility of a source is uncertain, clearly flag it.
- Do **not** treat information as trustworthy just because it appears — **cross-check when necessary**.
- If you find conflicting or ambiguous information, include all relevant findings and flag the inconsistency.

Be cautious and transparent in your output:
- Always return all related information. If information is incomplete or weakly supported, still share partial excerpts, and flag any uncertainty.
- Never assume or guess — if an exact answer cannot be found, say so clearly.
- Prefer quoting or excerpting **original source text** rather than interpreting or rewriting it, and provide the URL if available.
- If more context is needed, return a clarification request and do not proceed with tool use.
        """

    else:
        raise ValueError(f"Invalid agent type: {agent_type}")


def get_system_prompt(schema_text: str, agent_type: str = "main_agent") -> str:
    """Generate the system prompt for the agentic system.

    Args:
        schema_text: The schema text of the tools
        agent_type: The type of the agent

    Returns:
        System prompt for the main-agent and agent-tools (e.g. browsing-agent)
    """

    agentic_prompt = get_agentic_system_prompt(agent_type)

    system_prompt = """
In this environment you have access to a set of tools you can use to answer the user's question.

You only have access to the tools provided below. You can only use one tool per message, and will receive the result of that tool in the user's next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use. Today is: 2025-07-16

# Tool-Use Formatting Instructions

Tool-use is formatted using XML-style tags. The tool-use is enclosed in <use_mcp_tool></use_mcp_tool> and each parameter is similarly enclosed within its own set of tags.

The Model Context Protocol (MCP) connects to servers that provide additional tools and resources to extend your capabilities. You can use the server's tools via the `use_mcp_tool`.

Description:
Request to use a tool provided by a MCP server. Each MCP server can provide multiple tools with different capabilities. Tools have defined input schemas that specify required and optional parameters.

Parameters:
- server_name: (required) The name of the MCP server providing the tool
- tool_name: (required) The name of the tool to execute
- arguments: (required) A JSON object containing the tool's input parameters, following the tool's input schema, quotes within string must be properly escaped, ensure it's valid JSON
"""

    system_prompt += """   
Usage:
<use_mcp_tool>
<server_name>server name here</server_name>
<tool_name>tool name here</tool_name>
<arguments>
{
 "param1": "value1",
 "param2": "value2 \"escaped string\""
}
</arguments>
</use_mcp_tool>

Important Notes:
- Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.
- Always adhere to this format for the tool use to ensure proper parsing and execution.

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
"""

    system_prompt += """

{schema_text}

# General Objective

You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.

## Task Strategy

1. Analyze the user's request and set clear, achievable sub-goals. Prioritize these sub-goals in a logical order.
2. Start with a concise, numbered, step-by-step plan outlining how you will solve the task before taking any action.
3. Work through these sub-goals sequentially. After each step, adjust your plan as needed.
4. Use tools strategically to accomplish each sub-goal.
5. Revise earlier steps if new information emerges.

## Tool-Use Guidelines

1. Each step must involve a single tool call, unless the task is already solved.
2. Before each tool call:
   - Summarize what is known.
   - Identify what is missing.
   - Choose the most relevant tool.
   - Verify all required parameters.
3. All tool queries must include full context.
4. Avoid vague queries. Each call should retrieve actionable information.
5. Extract and summarize partial information if a tool result is incomplete.

## Tool-Use Communication Rules

1. Do not include tool results in your response.
2. Do not present the final answer until the entire task is complete.
3. Do not mention tool names.
4. Do not engage in unnecessary back-and-forth.
5. Do not use non-existent tools.
6. Respond in the same language as the user's message.
7. If the task does not require tool use, answer directly.

{agentic_prompt}

""".format(schema_text=schema_text, agentic_prompt=agentic_prompt)

    return system_prompt


def generate_agent_summarize_prompt(task_description, main_query=None, task_failed=False, agent_type=""):
    if agent_type == "main_agent":
        failed_instruction = ""
        if task_failed:
            failed_instruction = "You have failed to complete the task. Review the conversation and provide a final answer based on the information gathered. If a definitive answer cannot be determined, state that clearly."

        summarize_prompt = f"""
[SYSTEM]
This is a direct instruction to you. This is your final turn. You MUST NOT use any tools.
Your task is to provide a final, structured report summarizing the entire conversation and providing a definitive answer to the original task.

The original task was: "{task_description}"

[INSTRUCTIONS]

{failed_instruction}

Your final response MUST be a clear, complete, and structured report in markdown format.
Organize the content into logical sections with the following headings: `## Conclusion`, `## Supporting Information`, and `## Observations`.

- **CRITICAL**: Do NOT include any raw URLs.
- Your response should only contain factual, specific, and well-organized information based on the conversation.
- Do not include speculative filler, vague summaries, or conversational text.

Here is an example of the required format:

# Final Response: [Title summarizing the task]

## Conclusion:
[A concise summary of your findings and the final answer to the user's question. Bold key information. If the answer is a number or short phrase, provide it directly.]

## Supporting Information:
[Detailed supporting facts, data, or quotes from the conversation that support your conclusion. Use bullet points or numbered lists for clarity.]
- Source 1: Brief description of finding 1.
- Source 2: Brief description of finding 2.

## Observations:
[Any additional context, confidence level, or notes on how the final answer was derived from the conversation history.]
"""
    elif agent_type == "browsing_agent":
        failed_instruction = ""
        if task_failed:
            failed_instruction = "You have failed to complete the task. Do not attempt to answer the original task. Instead, clearly acknowledge that the task has failed and explain why in the 'Observations' section."

        summarize_prompt = f"""
[SYSTEM]
This is a direct instruction to you. This is your final turn. You MUST NOT use any tools.
Your task is to provide a final, structured report summarizing all the information you have gathered to answer your assigned subtask.

[CONTEXT]
The main task was: "{main_query}"
Your assigned subtask was: "{task_description}"
Your assigned subtask was intended to help solve the main task.

[INSTRUCTIONS]

{failed_instruction}

Your final response MUST be a clear, complete, and structured report in markdown format.
Organize the content into logical sections with the following headings: `## Conclusion`, `## Supporting Information`, `## Observations`, and `## Contribution to Main Task`.

- **CRITICAL**: Do NOT include raw URLs. Replace any URLs with `([link])`.
- Your response should only contain factual, specific, and well-organized information based on your previous actions.
- Do not include speculative filler, vague summaries, or conversational text.

Here is an example of the required format:

# Final Response: [Title summarizing the subtask]

## Conclusion:
[A concise summary of your findings and the final answer for the subtask. Bold key information.]

## Supporting Information:
[Detailed supporting facts, data, or quotes you discovered. Use bullet points or numbered lists for clarity.]
- Source 1: Brief description of finding 1.
- Source 2: Brief description of finding 2.

## Observations:
[Any additional context, confidence level, or notes on how the conclusion was reached.]

## Contribution to Main Task:
[Explain how the answer to your subtask helps solve the overall main task. What are the next steps the main agent should consider?]
"""
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return summarize_prompt
