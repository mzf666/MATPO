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
import unittest
from transformers import AutoTokenizer
from verl.workers.reward_manager.tool import ToolRewardManager

'''
Usage for running test case

    export OPENAI_API_KEY=xxx
    python test_mcp_format.py

'''

#########################################################
#####   Examples of response string (with bad cases) 
#########################################################

## gold class response string
RESPONSE_STR_GOLD_CLASS = '''
<think>

</think>

<use_mcp_tool>
<server_name>tool-google-search</server_name>
<tool_name>google_search</tool_name>
<arguments>
{
  "q": "IEEE Frank Rosenblatt Award 2010 recipient",
  "gl": "us",
  "hl": "en"
}
</arguments>
</use_mcp_tool>
user
(user content masked to simplify example of MCP tool calling response string)
'''

## BAD CASE 1: tool name is not given in system instruction -> ERROR: tool_`chrome`_search instead of `google`
RESPONSE_STR_WITH_UNKNOWN_TOOL_AND_SERVER_NAME = '''
<think>

</think>

<use_mcp_tool>
<server_name>tool-chrome-search</server_name>
<tool_name>chrome_search</tool_name>
<arguments>
{
  "q": "IEEE Frank Rosenblatt Award 2010 recipient",
  "gl": "us",
  "hl": "en"
}
</arguments>
</use_mcp_tool>
user
(user content masked to simplify example of MCP tool calling response string)
'''

## BAD CASE 2: missing or extra/typo argument given -> ERROR: missing `hl` and typo for location (without s)
RESPONSE_STR_WITH_MISSING_OR_EXTRA_ARGUMENT = '''
<think>

</think>

<use_mcp_tool>
<server_name>tool-google-search</server_name>
<tool_name>google_search</tool_name>
<arguments>
{
  "q": "IEEE Frank Rosenblatt Award 2010 recipient",
  "gl": "us",
  "locations": "New York and Chesapeake Bay, United States"
}
</arguments>
</use_mcp_tool>
user
(user content masked to simplify example of MCP tool calling response string)
'''

## BAD CASE 3: mismatch type from instruction arguments -> ERROR: args "num" should be `number` not `string`
RESPONSE_STR_WITH_MISMATCH_ARG_TYPE = '''
<think>

</think>

<use_mcp_tool>
<server_name>tool-google-search</server_name>
<tool_name>google_search</tool_name>
<arguments>
{
  "q": "IEEE Frank Rosenblatt Award 2010 recipient",
  "gl": "us",
  "hl": "en",
  "num": "10"
}
</arguments>
</use_mcp_tool>
user
(user content masked to simplify example of MCP tool calling response string)
'''

## BAD CASE 4: XML error format -> ERROR: not using closing tokens </arguments>, </use_mcp_tool>
RESPONSE_STR_WITH_XML_MISFORMAT = '''
<think>

</think>

<use_mcp_tool>
<server_name>tool-google-search</server_name>
<tool_name>google_search</tool_name>
<arguments>
{
  "q": "IEEE Frank Rosenblatt Award 2010 recipient",
  "gl": "us",
  "hl": "en"
}
<arguments>
<use_mcp_tool>
user
(user content masked to simplify example of MCP tool calling response string)
'''


#########################################################
## Examples of system prompt string for MCP instructions
#########################################################

PROMPT_STR = '''
system

In this environment you have access to a set of tools you can use to answer the user's question.

You only have access to the tools provided below. You can only use one tool per message, and will receive the result of that tool in the user's next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use. Today is: 2025-06-16

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
 "param2": "value2 "escaped string""
}
</arguments>
</use_mcp_tool>

Important Notes:
- Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.
- Always adhere to this format for the tool use to ensure proper parsing and execution.

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.

Here are the functions available in JSONSchema format:

## Server name: tool-google-search
### Tool name: google_search
Description: Tool to perform web searches via Serper API and retrieve rich results. It is able to retrieve organic search results, people also ask, related searches, and knowledge graph.
Input JSON schema: {
  'type': 'object',
  'properties': {
    'q': {'type': 'string', 'description': 'Search query string'},
    'gl': {'type': 'string', 'description': "Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')"},
    'hl': {'type': 'string', 'description': "Optional language code for search results in ISO 639-1 format (e.g., 'en')"},
    'location': {'type': 'string', 'description': "Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')"},
    'num': {'type': 'number', 'description': 'Number of results to return (default: 10)'},
    'tbs': {'type': 'string', 'description': "Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year)"},
    'page': {'type': 'number', 'description': 'Page number of results to return (default: 1)'},
    'autocorrect': {'type': 'boolean', 'description': 'Whether to autocorrect spelling in query'}
  },
  'required': ['q', 'gl', 'hl']
}

### Tool name: scrape
Description: Tool to scrape a webpage and retrieve the text and, optionally, the markdown content. It will retrieve also the JSON-LD metadata and the head metadata.
Input JSON schema: {
  'type': 'object',
  'properties': {
    'url': {'type': 'string', 'description': 'The URL of the webpage to scrape.'},
    'includeMarkdown': {'type': 'boolean', 'description': 'Whether to include markdown content.', 'default': False}
  },
  'required': ['url']
}

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

user
Who received the IEEE Frank Rosenblatt Award in 2010? You should follow the format instruction in the requestion strictly and wrap the final answer in \boxed{}.
assistant

'''

# --- Dummy initialize of tokenizer ---
class DummyTokenizer:
    eos_token_id = 0
    pad_token_id = 1


# --- Test Case ---
class TestToolRewardManager(unittest.TestCase):

    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.reward_manager = ToolRewardManager(
            tokenizer=self.tokenizer,
            num_examine=0,)

    # Verify with above 5 cases (Gold class, BAD CASE 1, BAD CASE 2, BAD CASE 3, BAD CASE 4)
    ## NOTE: using verbose = True to print the error in format
    def test_tool_format_gold_class(self):
        ## expected 1.0
        score = self.reward_manager.check_mcp_format(PROMPT_STR, RESPONSE_STR_GOLD_CLASS, verbose=True)
        self.assertEqual(score, 1.0)
        print("Case pass: Gold standard MCP format in response\n")

    def test_tool_format_bad_case1(self):
        ## expected 0.0
        score = self.reward_manager.check_mcp_format(PROMPT_STR, RESPONSE_STR_WITH_UNKNOWN_TOOL_AND_SERVER_NAME, verbose=True)
        self.assertEqual(score, 0.0)
        print("Case pass: Response string with unknown tool and server name\n")

    def test_tool_format_bad_case2(self):
        ## expected 0.0
        score = self.reward_manager.check_mcp_format(PROMPT_STR, RESPONSE_STR_WITH_MISSING_OR_EXTRA_ARGUMENT, verbose=True)
        self.assertEqual(score, 0.0)
        print("Case pass: Response string with missing or extra argument(s)\n")

    def test_tool_format_bad_case3(self):
        ## expected 0.0
        score = self.reward_manager.check_mcp_format(PROMPT_STR, RESPONSE_STR_WITH_MISMATCH_ARG_TYPE, verbose=True)
        self.assertEqual(score, 0.0)
        print("Case pass: Response string with mismatch argument types\n")

    def test_tool_format_bad_case4(self):
        ## expected 0.0 
        score = self.reward_manager.check_mcp_format(PROMPT_STR, RESPONSE_STR_WITH_XML_MISFORMAT, verbose=True)
        self.assertEqual(score, 0.0)
        print("Case pass: Response string with bad xml format\n")

# --- Entry point to run the tests ---

if __name__ == "__main__":
    unittest.main()