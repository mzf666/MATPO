import argparse
import asyncio
import os

import datasets

from examples.data_preprocess.prompt_utils import (
    format_tool_schemas_for_prompt,
    get_system_prompt,
    load_all_tool_schemas_and_kwargs,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data_path", type=str, required=True)
    parser.add_argument("--dst_data_path", type=str, required=True)
    parser.add_argument("--tool_config_path", type=str, 
                        default="examples/sglang_multiturn/config/tool_config/mcp_tool_config_full_agent.yaml",
                        help="Path to the agent-tool configuration YAML file.")
    args = parser.parse_args()

    assert os.path.exists(args.src_data_path), f"Source data path {args.src_data_path} does not exist."
    assert args.dst_data_path != args.src_data_path, "Destination data path cannot be the same as source data path."

    tool_schemas, base_tools_kwargs, agent_tools_map = asyncio.run(load_all_tool_schemas_and_kwargs(args.tool_config_path, is_main_agent=True))
    tools_filter = agent_tools_map["main_agent"]
    tool_schemas_text = format_tool_schemas_for_prompt(tool_schemas, tools_filter)
    system_prompt = get_system_prompt(tool_schemas_text, agent_type="main_agent")

    data = datasets.load_dataset("parquet", data_files=args.src_data_path)["train"]
    data_dict = data.to_dict()
    # print(data_dict["prompt"][0])

    prompts = data_dict["prompt"]
    extra_infos = data_dict["extra_info"]

    for i, (prompt, extra_info) in enumerate(zip(prompts, extra_infos)):
        assert prompt[0]["role"] == "system", "The first message in the prompt should be the system prompt."
        prompts[i][0]["content"] = system_prompt
        extra_infos[i]["tools_kwargs"] = base_tools_kwargs

    data_dict["prompt"] = prompts
    data_dict["extra_info"] = extra_infos

    dataset = datasets.Dataset.from_dict(data_dict)
    dataset.to_parquet(args.dst_data_path)
    print(f"\n {len(prompts)} data processed (change sys prompt & tools_kwargs) {args.src_data_path} --> {args.dst_data_path}")
