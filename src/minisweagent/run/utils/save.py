import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from minisweagent import Agent, __version__


def _get_class_name_with_module(obj: Any) -> str:
    """Get the full class name with module path."""
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


def save_traj(
    agent: Agent | None,
    path: Path | None,
    *,
    print_path: bool = True,
    exit_status: str | None = None,
    result: str | None = None,
    extra_info: dict | None = None,
    print_fct: Callable = print,
    **kwargs,
):
    """Save the trajectory of the agent to a file.

    Args:
        agent: The agent to save the trajectory of.
        path: The path to save the trajectory to.
        print_path: Whether to print confirmation of path to the terminal.
        exit_status: The exit status of the agent.
        result: The result/submission of the agent.
        extra_info: Extra information to save (will be merged into the info dict).
        **kwargs: Additional information to save (will be merged into top level)

    """
    if path is None:
        return
    data = {
        "info": {
            "exit_status": exit_status,
            "submission": result,
            "model_stats": {
                "instance_cost": 0.0,
                "api_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "reward_model_stats": {
                "instance_cost": 0.0,
                "api_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "mini_version": __version__,
        },
        "messages": [],
        "trajectory_format": "mini-swe-agent-1",
    } | kwargs
    if agent is not None:
        data["info"]["model_stats"]["instance_cost"] = agent.model.cost
        data["info"]["model_stats"]["api_calls"] = agent.model.n_calls
        data["info"]["model_stats"]["input_tokens"] = agent.model.input_tokens
        data["info"]["model_stats"]["output_tokens"] = agent.model.output_tokens
        if hasattr(agent, "reward_model"):
            data["info"]["reward_model_stats"]["instance_cost"] = agent.reward_model.model.cost
            data["info"]["reward_model_stats"]["api_calls"] = agent.reward_model.model.n_calls
            data["info"]["reward_model_stats"]["input_tokens"] = agent.reward_model.model.input_tokens
            data["info"]["reward_model_stats"]["output_tokens"] = agent.reward_model.model.output_tokens
        data["messages"] = agent.messages
        data["info"]["config"] = {
            "agent": agent.config.model_dump(),
            "model": agent.model.config.model_dump(),
            "reward_model": agent.reward_model.model.config.model_dump() if hasattr(agent, "reward_model") else None,
            "environment": agent.env.config.model_dump(),
            "agent_type": _get_class_name_with_module(agent),
            "model_type": _get_class_name_with_module(agent.model),
            "reward_model_type": _get_class_name_with_module(agent.reward_model.model) if hasattr(agent, "reward_model") else None,
            "environment_type": _get_class_name_with_module(agent.env),
        }
    if extra_info:
        data["info"].update(extra_info)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    if print_path:
        print_fct(f"Saved trajectory to '{path}'")
