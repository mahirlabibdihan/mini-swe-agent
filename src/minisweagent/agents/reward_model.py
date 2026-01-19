"""Reward model for evaluating actions in tree search."""

from typing import Any, Optional, Dict
from dataclasses import dataclass
import abc
from minisweagent.agents.tree_search_node import TreeSearchNode
from minisweagent import Model
import random

class RewardModel():
    def __init__(self, model: Model):
        self.model = model
    
    def compute_reward(
        self,
        node: TreeSearchNode,
        task: Optional[str] = None,
    ) -> float:
        """Compute reward for an action.
        
        Args:
            node: The current tree search node
            task: Optional task description for context
            
        Returns:
            A float reward value. Higher is better.
        """
        return random.random()  # A random number from 0 to 1 for now; replace with proper evaluation later

