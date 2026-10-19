"""GraphModel: the mutable world shared by A* and D* Lite.

Holds states, transitions (succ/pred adjacency), bad states, the distance of every state
to its nearest bad state, and the effective edge cost used for planning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

from models import PlanningProblem, PlanningResult, Transition

INF = math.inf


@dataclass
class Weights:
    w_safety: float = 2.0         # penalty for (1 - transition.safety)
    w_reliability: float = 1.0    # penalty for (1 - transition.reliability)
    w_proximity: float = 5.0      # penalty for entering a state close to a bad state
    d_safe: float = 2.0           # distance at which the proximity penalty reaches 0
    hard_margin: float = 0.0      # states with dist <= hard_margin are forbidden (0 = only bad states)


class GraphModel:
    def __init__(self, problem: PlanningProblem, weights: Optional[Weights] = None):
        self.w = weights or Weights()
        self.ids: List[int] = [s.id for s in problem.states]
        self.states: Dict[int, np.ndarray] = {
            s.id: np.asarray(s.embedding, dtype=float) for s in problem.states}
        self._X = np.array([self.states[i] for i in self.ids])      # (n, d) matrix
        self._pts = {i: tuple(self.states[i]) for i in self.ids}    # tuples: fast math.dist
        self.trans: Dict[int, Transition] = {}
        self.succ: Dict[int, Dict[int, Transition]] = {i: {} for i in self.ids}
        self.pred: Dict[int, Dict[int, Transition]] = {i: {} for i in self.ids}
        self._min_ratio = INF                       # min over edges of cost / euclid length
        for t in problem.transitions:
            self._insert(t)
        self.bad: Set[int] = set()
        self.bad_dist: Dict[int, float] = {i: INF for i in self.ids}
        self.set_bad_states(problem.bad_states)

    # ------------------------------------------------------------------ heuristic
    @property
    def scale(self) -> float:
        return self._min_ratio if math.isfinite(self._min_ratio) else 0.0

    def h(self, a: int, b: int) -> float:
        """Admissible + consistent: scale * Euclidean distance (scale = cheapest cost per unit length)."""
        return self.scale * math.dist(self._pts[a], self._pts[b])

    # ------------------------------------------------------------------ safety
    def set_bad_states(self, bad_ids: Iterable[int]) -> Set[int]:
        """Replace the bad-state set; returns ids whose distance-to-bad changed."""
        new_bad = set(bad_ids)
        for b in new_bad:
            if b not in self.states:
                raise KeyError(f"bad state {b} is not a known state")
        self.bad = new_bad
        if new_bad:
            B = np.array([self.states[b] for b in new_bad])
            n = len(self.ids)
            dist = np.empty(n)
            chunk = 512
            for lo in range(0, n, chunk):
                diff = self._X[lo:lo + chunk, None, :] - B[None, :, :]
                dist[lo:lo + chunk] = np.linalg.norm(diff, axis=2).min(axis=1)
            new_dist = {i: float(dist[k]) for k, i in enumerate(self.ids)}
        else:
            new_dist = {i: INF for i in self.ids}
        changed = {i for i in self.ids if new_dist[i] != self.bad_dist[i]}
        self.bad_dist = new_dist
        return changed

    def blocked(self, u: int) -> bool:
        return u in self.bad or self.bad_dist[u] <= self.w.hard_margin

    def edge_cost(self, t: Transition) -> float:
        """Effective cost = cost + safety/reliability/proximity penalties (INF if unusable)."""
        if (not t.available) or self.blocked(t.src) or self.blocked(t.dst):
            return INF
        d = self.bad_dist[t.dst]
        prox = max(0.0, 1.0 - d / self.w.d_safe) if math.isfinite(d) else 0.0
        return (t.cost
                + self.w.w_safety * (1.0 - t.safety)
                + self.w.w_reliability * (1.0 - t.reliability)
                + self.w.w_proximity * prox)

    # ------------------------------------------------------------------ mutation
    def _insert(self, t: Transition) -> bool:
        """Insert edge. Returns True if the heuristic scale decreased."""
        if t.src not in self.states or t.dst not in self.states:
            raise KeyError("transition refers to unknown state")
        if t.cost < 0:
            raise ValueError("negative transition costs are not supported")
        self.trans[t.id] = t
        self.succ[t.src][t.id] = t
        self.pred[t.dst][t.id] = t
        length = float(np.linalg.norm(self.states[t.src] - self.states[t.dst]))
        if length > 1e-12 and t.cost / length < self._min_ratio:
            self._min_ratio = t.cost / length
            return True
        return False

    def add_transition(self, t: Transition) -> bool:
        return self._insert(t)

    def remove_transition(self, tid: int) -> int:
        t = self.trans.pop(tid)
        del self.succ[t.src][tid]
        del self.pred[t.dst][tid]
        return t.src

    def set_available(self, tid: int, flag: bool) -> int:
        t = self.trans[tid]
        t.available = flag
        return t.src


def make_result(model: GraphModel, state_path: Optional[List[int]],
                trans_path: Optional[List[int]], expanded: int, elapsed: float,
                message: str = "") -> PlanningResult:
    if not state_path:
        return PlanningResult(success=False, expanded=expanded, planning_time=elapsed,
                              message=message or "no safe path found")
    ts = [model.trans[i] for i in trans_path]
    return PlanningResult(
        success=True,
        state_path=state_path,
        transition_path=trans_path,
        total_cost=sum(t.cost for t in ts),
        effective_cost=sum(model.edge_cost(t) for t in ts),
        safety_score=min(model.bad_dist[s] for s in state_path),
        total_reliability=sum(t.reliability for t in ts),
        bad_states_visited=sum(1 for s in state_path if s in model.bad),
        expanded=expanded,
        planning_time=elapsed,
        message=message or "ok",
    )
