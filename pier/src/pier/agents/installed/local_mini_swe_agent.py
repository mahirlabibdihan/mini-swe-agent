"""Adapters for the mini-swe-agent fork vendored next to this Pier checkout."""

from __future__ import annotations

import base64
import hashlib
import io
import shlex
import tarfile
from pathlib import Path

from pier.agents.installed.base import with_prompt_template
from pier.agents.installed.mini_swe_agent import MiniSweAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist
from pier.models.trial.paths import EnvironmentPaths


LOCAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


def _local_source_archive() -> bytes:
    """Archive the local agent source without outputs, tasks, or Git metadata."""
    buffer = io.BytesIO()

    def exclude_generated(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if "__pycache__" in parts or info.name.endswith((".pyc", ".pyo")):
            return None
        return info

    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for filename in ("pyproject.toml", "README.md", "LICENSE.md"):
            archive.add(
                LOCAL_REPOSITORY_ROOT / filename,
                arcname=filename,
                filter=exclude_generated,
            )
        archive.add(
            LOCAL_REPOSITORY_ROOT / "src",
            arcname="src",
            filter=exclude_generated,
        )
    return buffer.getvalue()


class ForkMiniSweAgent(MiniSweAgent):
    """Common installation and OpenRouter behavior for the local fork."""

    def install_spec(self) -> AgentInstallSpec:
        spec = super().install_spec()
        spec.agent_name = self.name()
        source_archive = _local_source_archive()
        encoded = base64.b64encode(source_archive).decode()
        unpack = (
            "mkdir -p /tmp/local-mini-swe-agent && "
            f"printf %s {shlex.quote(encoded)} | base64 -d | "
            "tar -xz -C /tmp/local-mini-swe-agent && "
        )
        replacement = f"{unpack}uv tool install /tmp/local-mini-swe-agent"
        for step in spec.steps:
            step.run = step.run.replace("uv tool install mini-swe-agent", replacement)
        digest = hashlib.sha256(source_archive).hexdigest()[:16]
        spec.cache_key = f"{self.name()}-local-{digest}"
        return spec

    def network_allowlist(self) -> NetworkAllowlist:
        parent = super().network_allowlist()
        return NetworkAllowlist(
            domains=parent.domains
            + [
                "openrouter.ai",
                "raw.githubusercontent.com",
                "astral.sh",
                "pypi.org",
                "files.pythonhosted.org",
            ]
        )

    @property
    def _fork_model_name(self) -> str:
        return (self.model_name or "").removeprefix("openrouter/")

    def _fork_env(self) -> dict[str, str]:
        env = self.build_process_env(
            {
                "MSWEA_CONFIGURED": "true",
                "MSWEA_COST_TRACKING": "ignore_errors",
                "LITELLM_LOCAL_MODEL_COST_MAP": "true",
            }
        )
        if key := self._get_env("OPENROUTER_API_KEY"):
            env["OPENROUTER_API_KEY"] = key
        else:
            raise ValueError("OPENROUTER_API_KEY is required")
        return env


class LocalMiniSweAgent(ForkMiniSweAgent):
    """The fork's normal/base InteractiveAgent."""

    @staticmethod
    def name() -> str:
        return "local-mini-swe-agent"

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if "/" not in self._fork_model_name:
            raise ValueError("Model name must be openrouter/provider/model")
        command = (
            '. "$HOME/.local/bin/env"; cd /app; '
            f"mini --yolo --exit-immediately --model {shlex.quote(self._fork_model_name)} "
            f"--model-class openrouter --task {shlex.quote(instruction)} "
            f"--output {self._mini_swe_agent_trajectory_path}; "
            "git config user.name 'DeepSWE Agent'; "
            "git config user.email 'agent@localhost'; git add -A; "
            "git diff --cached --quiet || git commit -m 'Agent solution'"
        )
        await self.exec_as_agent(environment, command=command, env=self._fork_env())


class TreeSearchMiniSweAgent(ForkMiniSweAgent):
    """The fork's tree-search InteractiveAgent."""

    @staticmethod
    def name() -> str:
        return "tree-search-mini-swe-agent"

    def install_spec(self) -> AgentInstallSpec:
        spec = super().install_spec()
        runner_path = (
            Path(__file__).resolve().parents[5] / "deep-swe" / "run_tree_search.py"
        )
        encoded = base64.b64encode(runner_path.read_bytes()).decode()
        spec.steps.append(
            InstallStep(
                user="agent",
                run=f"printf %s {shlex.quote(encoded)} | base64 -d > /tmp/run_tree_search.py",
            )
        )
        return spec

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if "/" not in self._fork_model_name:
            raise ValueError("Model name must be openrouter/provider/model")
        encoded = base64.b64encode(instruction.encode()).decode()
        command = (
            f"printf %s {shlex.quote(encoded)} | base64 -d > /tmp/deep-swe-instruction.md; "
            '. "$HOME/.local/bin/env"; '
            "python /tmp/run_tree_search.py --task-file /tmp/deep-swe-instruction.md "
            f"--model {shlex.quote(self._fork_model_name)} --cwd /app "
            f"--output {EnvironmentPaths.agent_dir}/mini-swe-agent.trajectory.json; "
            "cd /app; git config user.name 'DeepSWE Agent'; "
            "git config user.email 'agent@localhost'; git add -A; "
            "git diff --cached --quiet || git commit -m 'Agent solution'"
        )
        await self.exec_as_agent(environment, command=command, env=self._fork_env())
