"""Pier installed-agent adapter for this repository's tree-search agent."""

import base64
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths


REPOSITORY = "https://github.com/mahirlabibdihan/mini-swe-agent.git"
REVISION = "c7922248868225aa10cd73a1d3c804a6827c035b"


class TreeSearchMiniSweAgent(BaseInstalledAgent):
    """Install the pinned fork and execute its headless tree-search runner."""

    @staticmethod
    def name() -> str:
        return "tree-search-mini-swe-agent"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl git build-essential",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=(
                "curl -LsSf https://astral.sh/uv/install.sh | sh && "
                f"$HOME/.local/bin/uv tool install 'git+{REPOSITORY}@{REVISION}' --with 'litellm[proxy]'"
            ),
        )
        runner = Path(__file__).with_name("run_tree_search.py").read_bytes()
        encoded = base64.b64encode(runner).decode()
        await self.exec_as_agent(
            environment,
            command=f"printf %s {shlex.quote(encoded)} | base64 -d > /tmp/run_tree_search.py",
        )

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if not self.model_name:
            raise ValueError("Pass Pier --model provider/model")
        task = base64.b64encode(instruction.encode()).decode()
        env = {name: value for name in (
            "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY", "MSWEA_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL",
        ) if (value := self._get_env(name)) is not None}
        env["MSWEA_CONFIGURED"] = "true"
        env["MSWEA_COST_TRACKING"] = "ignore_errors"
        command = (
            f"printf %s {shlex.quote(task)} | base64 -d > /tmp/deep-swe-instruction.md && "
            ". $HOME/.local/bin/env && "
            f"python /tmp/run_tree_search.py --task-file /tmp/deep-swe-instruction.md "
            f"--model {shlex.quote(self.model_name)} --cwd /app "
            f"--output {EnvironmentPaths.agent_dir}/mini-swe-agent.trajectory.json && "
            "cd /app && git config user.name 'DeepSWE Agent' && "
            "git config user.email 'agent@localhost' && git add -A && "
            "(git diff --cached --quiet || git commit -m 'Agent solution')"
        )
        await self.exec_as_agent(environment, command=command, env=env)

    def populate_context_post_run(self, context: AgentContext) -> None:
        # The native mini-swe-agent trajectory remains in the trial artifacts.
        return None


class LocalMiniSweAgent(BaseInstalledAgent):
    """Run the normal/base agent from this repository's pinned revision."""

    @staticmethod
    def name() -> str:
        return "local-mini-swe-agent"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl git build-essential",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=(
                "curl -LsSf https://astral.sh/uv/install.sh | sh && "
                f"$HOME/.local/bin/uv tool install 'git+{REPOSITORY}@{REVISION}' --with 'litellm[proxy]'"
            ),
        )

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if not self.model_name:
            raise ValueError("Pass Pier --model provider/model")
        task = base64.b64encode(instruction.encode()).decode()
        env = {name: value for name in (
            "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY", "MSWEA_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL",
        ) if (value := self._get_env(name)) is not None}
        env["MSWEA_CONFIGURED"] = "true"
        env["MSWEA_COST_TRACKING"] = "ignore_errors"
        command = (
            f"printf %s {shlex.quote(task)} | base64 -d > /tmp/deep-swe-instruction.md && "
            ". $HOME/.local/bin/env && cd /app && "
            f"mini --yolo --exit-immediately --model {shlex.quote(self.model_name)} "
            "--model-class openrouter --task \"$(cat /tmp/deep-swe-instruction.md)\" "
            f"--output {EnvironmentPaths.agent_dir}/mini-swe-agent.trajectory.json && "
            "git config user.name 'DeepSWE Agent' && "
            "git config user.email 'agent@localhost' && git add -A && "
            "(git diff --cached --quiet || git commit -m 'Agent solution')"
        )
        await self.exec_as_agent(environment, command=command, env=env)

    def populate_context_post_run(self, context: AgentContext) -> None:
        # The native mini-swe-agent trajectory remains in the trial artifacts.
        return None
