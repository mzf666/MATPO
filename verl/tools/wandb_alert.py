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
import time
from typing import Dict, Optional, Tuple
import re

import ray
import wandb
from wandb import AlertLevel


def get_or_init_wandb():
    runtime_env = ray.get_runtime_context().runtime_env
    env_vars = runtime_env.get("env_vars", {})

    if env_vars.get("ENABLE_WANDB_ALERT", "False") == "False" or os.environ.get("WANDB_API_KEY") is None:
        return None

    if wandb.run is None:
        wandb.init(project=env_vars.get("WANDB_PROJECT"), name=env_vars.get("WANDB_NAME"))
    return wandb.run


wandb_run = get_or_init_wandb()


@ray.remote(num_cpus=1)
class MCPToolChecker:
    """Monitor for tracking tool calling error statistics and triggering alerts."""
    
    def __init__(self, config):
        """
        Initialize error monitor.

        Args:
            alert_threshold: Number of errors of the same type before alerting
            monitoring_window: Time window in seconds for error counting
        """
        self.alert_threshold = config.trainer.get("tool_check_alert_threshold", 10)
        # key: iteration, value: dict of error type and error count
        self.error_counts: Dict[int, dict[str, int]] = {}
        # key: iteration, value: total error counts
        self.total_error_counts: Dict[int, int] = {}

        self.latest_iteration = 1

        if os.environ.get("WANDB_API_KEY") is not None:
            self.wandb_run = wandb.init(project=config["trainer"]["project_name"], name=config["trainer"]["experiment_name"])
        else:
            self.wandb_run = None

    def update_latest_iteration(self, iteration: int) -> int:
        """
        Update the latest iteration.
        Check if alert should be triggered.
        """
        if self.latest_iteration > 1:
            if self.latest_iteration not in self.total_error_counts:
                cur_error_counts = 0
            else:
                cur_error_counts = self.total_error_counts[self.latest_iteration]
            
            if self.latest_iteration - 1 not in self.total_error_counts:
                pre_error_counts = 0
            else:
                pre_error_counts = self.total_error_counts[self.latest_iteration - 1]

            # alert condition:
            if cur_error_counts > pre_error_counts * self.alert_threshold:
                print(f"MCPToolChecker Step {self.latest_iteration} Alert: {cur_error_counts=} >= {pre_error_counts=} * {self.alert_threshold}", flush=True)
                if self.wandb_run is not None:
                    self.wandb_run.alert(
                        title="MCP Tool Call Failed",
                        text=f"MCP Tool Call failed total times: {cur_error_counts}, Detail Error Type: {self.error_counts[self.latest_iteration]}",
                        level=AlertLevel.WARN,
                    )

        self.latest_iteration = iteration
        return self.latest_iteration

    def is_timeout_error(self, error_msg: Exception) -> bool:
        """Check if an error is a timeout-related error."""
        error_str = str(error_msg)
        return any(keyword in error_str for keyword in ["ETIMEDOUT", "ECONNRESET", "Timeout", "Timed out"])

    def extract_error_code(self, error_msg: Exception) -> Optional[str]:
        """Extract error code from MCP error or other exceptions."""
        if self.is_timeout_error(error_msg):
            return "TIMEOUT"

        error_code = str(error_msg)

        # Try to extract API error codes from error responses
        api_code_match = re.search(r'"statusCode":\s*(\d+)', error_code)
        if api_code_match:
            # 400: Bad Request
            # 500: Internal Server Error
            # 404: Not Found
            # 429: Too Many Requests
            error_code = api_code_match.group(1)
        # Try to extract specific error types from error messages
        elif "503" in error_code:
            # 503: Service Unavailable
            error_code = "503"
        elif "socket hang up" in error_code:
            error_code = "socket hang up"
        elif "Search query and region code and language are required" in error_code:
            # skip this error
            error_code = None

        return error_code

    def record_error(self, error_msg: Exception, tool_name: str) -> bool:
        """
        Record an error.

        Args:
            error_code: The error code/type
            tool_name: Name of the tool that failed
            
        Returns:
            True if error is recorded, False otherwise
        """

        error_code = self.extract_error_code(error_msg)
        if error_code is None:
            return False

        error_type = f"{tool_name} {error_code}"

        if self.latest_iteration not in self.total_error_counts:
            self.total_error_counts[self.latest_iteration] = 0

        if self.latest_iteration not in self.error_counts:
            self.error_counts[self.latest_iteration] = {}

        if error_type not in self.error_counts[self.latest_iteration]:
            self.error_counts[self.latest_iteration][error_type] = 0

        self.error_counts[self.latest_iteration][error_type] += 1
        self.total_error_counts[self.latest_iteration] += 1

        return True


@ray.remote(num_cpus=1)
class HangChecker:
    """Check if the training is hang."""
    def __init__(self, config):
        self.latest_iteration_path = os.path.join(config.trainer.default_local_dir, "latest_iteration.txt")
        self.check_interval = config.trainer.get("hang_check_interval", 3600)
        self.prev_latest_iteration = 0

        self.reset_latest_iteration()

        if os.environ.get("WANDB_API_KEY") is not None:
            self.wandb_run = wandb.init(project=config["trainer"]["project_name"], name=config["trainer"]["experiment_name"])
        else:
            self.wandb_run = None

    def get_latest_iteration(self):
        if os.path.exists(self.latest_iteration_path):
            with open(self.latest_iteration_path, "r") as f:
                return int(f.read())
        return 0

    def reset_latest_iteration(self):
        if not os.path.exists(self.latest_iteration_path):
            os.makedirs(os.path.dirname(self.latest_iteration_path), exist_ok=True)

        with open(self.latest_iteration_path, "w") as f:
            f.write("0")

    def check_hang_and_alert(self):
        while True:
            time.sleep(self.check_interval)
            latest_iteration = self.get_latest_iteration()
            if not (latest_iteration > self.prev_latest_iteration):
                if self.wandb_run is not None:
                    self.wandb_run.alert(
                        title="Training Hang",
                        text=f"Global step is not updated for {self.check_interval} seconds, maybe training is hang",
                        level=AlertLevel.WARN,
                    )
            self.prev_latest_iteration = latest_iteration
