"""Headless entry point used by the Pier adapter inside a DeepSWE container."""

import argparse
import traceback
from pathlib import Path

import yaml

from minisweagent.agents.interactive_ts import InteractiveAgent
from minisweagent.agents.reward_model import RewardModel
from minisweagent.config import get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="extra/swebench_ts.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cwd", default="/app")
    args = parser.parse_args()

    config = yaml.safe_load(get_config_path(Path(args.config)).read_text())
    environment_config = config.get("environment", {})
    environment_config.pop("environment_class", None)
    environment_config["cwd"] = args.cwd
    environment = LocalEnvironment(**environment_config)
    reward_config = config.get("reward_model", {})
    agent = InteractiveAgent(
        get_model(args.model, config.get("model", {})),
        environment,
        RewardModel(
            get_model(None, reward_config),
            use_combined_scoring=reward_config.get("use_combined_scoring", True),
            max_retries=reward_config.get("max_retries", 3),
        ),
        **({"mode": "yolo", "confirm_exit": False} | config.get("agent", {})),
    )

    exit_status = result = extra_info = None
    try:
        exit_status, result = agent.run(Path(args.task_file).read_text())
    except Exception as exc:
        exit_status, result = type(exc).__name__, str(exc)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(agent, Path(args.output), exit_status=exit_status, result=result, extra_info=extra_info)


if __name__ == "__main__":
    main()
