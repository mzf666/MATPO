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

import requests
from fastmcp import FastMCP
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize FastMCP server
mcp = FastMCP("audio-mcp-server")


@mcp.tool()
async def audio_transcription(audio_path_or_url: str) -> str:
    """
    Transcribe audio file to text and return the transcription.
    Args:
        audio_path_or_url: The path of the audio file locally or its URL.

    Returns:
        The transcription of the audio file.
    """

    client = OpenAI(api_key=OPENAI_API_KEY)

    if os.path.exists(audio_path_or_url):  # Check if the file exists locally
        audio_file = open(audio_path_or_url, "rb")
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe", file=audio_file
        )
    else:
        # download the audio file from the URL
        response = requests.get(audio_path_or_url)
        temp_audio_path = "/tmp/temp_audio_file"
        with open(temp_audio_path, "wb") as f:
            f.write(response.content)

        audio_file = open(temp_audio_path, "rb")
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe", file=audio_file
        )
        # Clean up the temp file
        audio_file.close()
        os.remove(temp_audio_path)

    return transcription.text


if __name__ == "__main__":
    mcp.run(transport="stdio")
