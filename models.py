"""Core data classes (Python equivalents of the C++ interfaces in the assignment)."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class State:
    id: int
    embedding: List[float]            # point in R^d


@dataclass
class Transition:
    id: int
    src: int                          # "from" in the C++ interface
    dst: int                          # "to"
    cost: float
    safety: float = 1.0               # in [0, 1], 1 = perfectly safe
    reliability: float = 1.0          # in [0, 1], 1 = perfectly reliable
    available: bool = True


@dataclass
class PlanningProblem:
    initial_state: int
    goal_state: int
    bad_states: List[int]
    states: List[State]
    transitions: List[Transition]


@dataclass
class PlanningResult:
    success: bool = False
    state_path: List[int] = field(default_factory=list)
    transition_path: List[int] = field(default_factory=list)
    total_cost: float = math.inf          # sum of raw transition costs
    effective_cost: float = math.inf      # cost + safety/reliability/proximity penalties
    safety_score: float = math.inf        # min Euclidean distance path -> nearest bad state
    total_reliability: float = 0.0        # sum of reliabilities along the path
    bad_states_visited: int = 0
    expanded: int = 0                     # states expanded during THIS search call
    planning_time: float = 0.0            # seconds
    message: str = ""


class Planner(ABC):
    @abstractmethod
    def plan(self, problem: PlanningProblem) -> PlanningResult:
        ...


def score(result: PlanningResult, alpha=1000.0, beta=1.0, gamma=10.0, delta=1.0,
          d_cap=10.0) -> float:
    """Score(P) = alpha*G - beta*C + gamma*D + delta*R  (D is capped at d_cap)."""
    if not result.success:
        return 0.0
    d = min(result.safety_score, d_cap)
    return alpha * 1.0 - beta * result.total_cost + gamma * d + delta * result.total_reliability
