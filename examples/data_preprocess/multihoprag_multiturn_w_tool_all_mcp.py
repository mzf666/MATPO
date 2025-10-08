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
from typing import Dict, List, Any
import time
import datasets
from omegaconf import OmegaConf

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
        args=config["args"] if "args" in config else [],
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
                        tools_schema.append({
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                            "server_name": config["server_name"],
                        })
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay_between_retries)
                    else:
                        raise
    return tools_schema


def load_tool_configs(config_path: str) -> Dict[str, Any]:
    """Load tool configurations from YAML file.
    
    Args:
        config_path: Path to the tool configuration file
        
    Returns:
        Dictionary containing tool configurations and schemas
    """
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


async def generate_tool_schemas_and_kwargs(config_path: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate tool schemas and kwargs from MCP tool configuration.
    
    Args:
        config_path: Path to the tool configuration file
        
    Returns:
        Tuple of (tool_schemas, tools_kwargs)
    """
    tools_config = load_tool_configs(config_path)
    all_tool_schemas = []
    all_tools_kwargs = {}
    
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
    
    return all_tool_schemas, all_tools_kwargs


def format_tool_schemas_for_prompt(tool_schemas: List[Dict[str, Any]]) -> str:
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
        if schema.get("server_name") != last_name:
            schema_text += f"## Server name: {schema['server_name']}\n"
            last_name = schema.get("server_name")
        schema_text += f"### Tool name: {schema['name']}\n"
        schema_text += f"Description: {schema['description']}\n"
        schema_text += f"Input JSON schema: {schema['parameters']}\n\n"
    
    return schema_text




def main(args):
    data_source = "yixuantt/MultiHopRAG"
    dataset = datasets.load_dataset(data_source, 'MultiHopRAG')

    test_dataset = dataset['train']

    instruction_following = "You should follow the format instruction in the requestion strictly and wrap the final answer in \\boxed{}."

    # Load tool schemas and kwargs
    tool_schemas, base_tools_kwargs = asyncio.run(generate_tool_schemas_and_kwargs(args.tool_config_path))
    tool_schemas_text = format_tool_schemas_for_prompt(tool_schemas)

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("query")
            question = question_raw + " " + instruction_following
            date=str(time.strftime("%Y-%m-%d", time.localtime()))
            answer_raw = example.pop("answer")
            answer = extract_answer(answer_raw)

            # Create tools_kwargs with ground_truth for this example
            tools_kwargs = {}
            for tool_name, tool_config in base_tools_kwargs.items():
                tools_kwargs[tool_name] = {
                    "create_kwargs": {"ground_truth": answer}
                }

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": (
"""
In this environment you have access to a set of tools you can use to answer the user's question.

You only have access to the tools provided below. You can only use one tool per message, and will receive the result of that tool in the user's next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use. Today is: """ + date + """

# Tool-Use Formatting Instructions

Tool-use is formatted using XML-style tags. The tool-use is enclosed in <use_mcp_tool></use_mcp_tool> and each parameter is similarly enclosed within its own set of tags.

The Model Context Protocol (MCP) connects to servers that provide additional tools and resources to extend your capabilities. You can use the server's tools via the `use_mcp_tool`.

Description:
Request to use a tool provided by a MCP server. Each MCP server can provide multiple tools with different capabilities. Tools have defined input schemas that specify required and optional parameters.

Parameters:
- server_name: (required) The name of the MCP server providing the tool
- tool_name: (required) The name of the tool to execute
- arguments: (required) A JSON object containing the tool's input parameters, following the tool's input schema, quotes within string must be properly escaped, ensure it's valid JSON

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
+
tool_schemas_text
+
"""

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

# Agent Specific Objective

You are a task-solving agent that uses tools step-by-step to answer the user's question. Your goal is to provide complete, accurate and well-reasoned answers using additional tools.
""")
                        ,
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

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Convert train_dataset to datasets.Dataset type
    train_dataset = datasets.Dataset.from_dict({
        k: test_dataset[k][500:] for k in test_dataset.features
    })
    test_dataset = datasets.Dataset.from_dict({
        k: test_dataset[k][:500] for k in test_dataset.features
    })


    local_dir = args.local_dir
    local_dir = os.path.expanduser(local_dir)
    local_dir = os.path.abspath(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
    
    print(f"Data saved to {local_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/multihoprag_dynamic_mcp")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--tool_config_path", default="examples/sglang_multiturn/config/tool_config/mcp_tool_config_full.yaml")
    
    args = parser.parse_args()
    main(args)
