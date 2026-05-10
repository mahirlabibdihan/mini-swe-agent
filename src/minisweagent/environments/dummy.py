import os
import platform
import subprocess
from typing import Any

from pydantic import BaseModel


class DummyEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30
    checkpoint: str = None # Path to the checkpoint file that the agent can use to simulate the tree search without actually executing any command or calculating reward. This is useful for testing and debugging the agent without consuming resources or affecting the local environment. The checkpoint file should be a .tree.json file that contains the tree structure and values from a previous run of the agent on the same instance. The agent can use this file to look up the values of nodes in the tree and simulate the merging process without actually executing any commands or calculating rewards. This allows us to test and debug the agent's logic and behavior without consuming resources or affecting the local environment.
    image: str = None # Optional image name for the dummy environment, for testing purposes. This can be used to simulate different environments or configurations without actually running a container. The agent can use this information to adjust its behavior or logic based on the specified image, allowing us to test how the agent would perform in different environments without needing to set up those environments.
    clean_start: bool = False # Whether to start with a clean environment (e.g., by resetting any state or data) for each run of the agent. This can be useful for testing and debugging the agent's behavior in a consistent and controlled environment, without any leftover state or data from previous runs affecting the results. If set to True, the dummy environment can simulate a clean start by resetting any relevant state or data before each run of the agent, allowing us to test the agent's behavior in a fresh environment each time.
    
class DummyEnvironment:
    def __init__(self, *, config_class: type = DummyEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        self.config = config_class(**kwargs)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None):
        """Execute a command in the local environment and return the result as a dict."""
        return {"output": "Dummy output", "returncode": 0}

    def get_template_vars(self) -> dict[str, Any]:
        return self.config.model_dump() | platform.uname()._asdict() | os.environ
