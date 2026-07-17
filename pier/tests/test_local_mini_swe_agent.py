from pathlib import Path

from pier.agents.factory import AgentFactory
from pier.models.agent.name import AgentName


def test_local_agents_are_registered(tmp_path: Path) -> None:
    base = AgentFactory.create_agent_from_name(
        AgentName.LOCAL_MINI_SWE_AGENT,
        logs_dir=tmp_path,
        model_name="openrouter/openai/gpt-5-mini",
        extra_env={"OPENROUTER_API_KEY": "test"},
    )
    tree = AgentFactory.create_agent_from_name(
        AgentName.TREE_SEARCH_MINI_SWE_AGENT,
        logs_dir=tmp_path,
        model_name="openrouter/openai/gpt-5-mini",
        extra_env={"OPENROUTER_API_KEY": "test"},
    )
    assert base.name() == "local-mini-swe-agent"
    assert tree.name() == "tree-search-mini-swe-agent"
