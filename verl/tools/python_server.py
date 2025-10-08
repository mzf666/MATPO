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

from e2b_code_interpreter import Sandbox
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("e2b-python-interpreter")

# API keys
E2B_API_KEY = os.environ.get("E2B_API_KEY")

# DEFAULT CONFS
DEFAULT_TIMEOUT = 300  # seconds


@mcp.tool()
async def run_python_code(
    code_block, timeout: int = DEFAULT_TIMEOUT, sandbox_id=None
) -> str:
    """Run python code in an interperter and return the execution result.

    Args:
        code_block: The python code to run.
        timeout: Time in seconds before the sandbox is automatically shutdown. The default is 300 seconds.
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. Only create new ones if this is the first time running code in this sandbox.

    Returns:
        An object containing the sandbox id and the execution result object including results, logs and errors.
    """
    if sandbox_id:
        sandbox = Sandbox.connect(sandbox_id, api_key=E2B_API_KEY)
    else:
        sandbox = Sandbox(timeout=timeout, api_key=E2B_API_KEY)

    execution = sandbox.run_code(code_block)

    sandbox_id = sandbox.get_info().sandbox_id

    return dict(execution_result=execution, sandbox_id=sandbox_id)


@mcp.tool()
async def upload_local_file_to_python_interpreter(
    local_file_path: str, sandbox_id=None
) -> str:
    """Upload a local file to the `/home/user` dir of the remote python interpreter.

    Args:
        file_path: The path of the file on local machine to upload.
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. Only create new ones if this is the first time running code in this sandbox.

    Returns:
        The path of the uploaded file in the remote python interpreter.
    """
    if sandbox_id:
        sandbox = Sandbox.connect(sandbox_id, api_key=E2B_API_KEY)
    else:
        sandbox = Sandbox(api_key=E2B_API_KEY)

    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"File {local_file_path} does not exist.")

    # Get the uploaded file path
    uploaded_file_path = os.path.join("/home/user", os.path.basename(local_file_path))

    # Upload the file
    with open(local_file_path, "rb") as f:
        sandbox.files.write(uploaded_file_path, f)

    sandbox_id = sandbox.get_info().sandbox_id

    return dict(uploaded_file_path=uploaded_file_path, sandbox_id=sandbox_id)


@mcp.tool()
async def download_internet_file_to_python_interpreter(
    url: str, sandbox_id=None
) -> str:
    """Download a file from the internet to the `/home/user` dir of the remote python interpreter.

    Args:
        url: The URL of the file to download.
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. Only create new ones if this is the first time running code in this sandbox.

    Returns:
        The path of the downloaded file in the python interpreter.
    """
    if sandbox_id:
        sandbox = Sandbox.connect(sandbox_id, api_key=E2B_API_KEY)
    else:
        sandbox = Sandbox(api_key=E2B_API_KEY)

    downloaded_file_path = os.path.join("/home/user", os.path.basename(url))

    # Download the file
    sandbox.commands.run(f"wget {url} -O {downloaded_file_path}")

    sandbox_id = sandbox.get_info().sandbox_id

    return dict(downloaded_file_path=downloaded_file_path, sandbox_id=sandbox_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
