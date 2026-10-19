"""D* Lite (Koenig & Likhachev, 2002) for safe planning with dynamic updates.

The search runs backwards from the goal, so g(s) = cost-to-goal. When edges change we only
repair the affected vertices instead of searching from scratch.
"""
from __future__ import annotations

import heapq
import itertools
import time
from typing import Iterable, Optional, Set

from graph import GraphModel, INF, Weights, make_result
from models import Planner, PlanningProblem, PlanningResult, Transition


class DStarLitePlanner(Planner):
    def __init__(self, weights: Optional[Weights] = None):
        self.weights = weights or Weights()
        self.model: Optional[GraphModel] = None

    # ================================================================ public API
    def plan(self, problem: PlanningProblem) -> PlanningResult:
        self.model = GraphModel(problem, self.weights)
        self.start = problem.initial_state
        self.goal = problem.goal_state
        self._reset_search()
        return self.replan()

    def replan(self) -> PlanningResult:
        """Repair the search after any pending updates and extract the current best path."""
        assert self.model is not None, "call plan() first"
        t0 = time.perf_counter()
        self.expanded = 0
        if self._needs_reset:
            self._reset_search()
        for u in self._dirty:
            self._update_vertex(u)
        self._dirty.clear()
        m = self.model
        if m.blocked(self.start) or m.blocked(self.goal):
            return make_result(m, None, None, 0, time.perf_counter() - t0,
                               "start or goal is a bad state")
        self._compute_shortest_path()
        spath, tpath = self._extract_path()
        return make_result(m, spath, tpath, self.expanded, time.perf_counter() - t0)

    # ---- dynamic-environment updates (call replan() afterwards) ----
    def set_goal(self, new_goal: int) -> None:
        """Goal change: graph, distance-to-bad table and edge costs are KEPT; only g/rhs reset."""
        self.goal = new_goal
        self._needs_reset = True

    def move_start(self, new_start: int) -> None:
        """The agent moved along the path; keeps all g/rhs values (uses the km offset)."""
        self.km += self.model.h(self._s_last, new_start)
        self._s_last = new_start
        self.start = new_start

    def set_bad_states(self, bad_ids: Iterable[int]) -> None:
        for n in self.model.set_bad_states(bad_ids):
            self._dirty.add(n)
            for t in self.model.pred[n].values():
                self._dirty.add(t.src)

    def set_transition_available(self, tid: int, flag: bool) -> None:
        self._dirty.add(self.model.set_available(tid, flag))

    def remove_transition(self, tid: int) -> None:
        self._dirty.add(self.model.remove_transition(tid))

    def add_transition(self, t: Transition) -> None:
        if self.model.add_transition(t):      # heuristic scale dropped -> old keys invalid
            self._needs_reset = True
        self._dirty.add(t.src)

    # ================================================================ internals
    def _reset_search(self) -> None:
        self.g, self.rhs = {}, {}
        self._heap, self._open = [], {}
        self._counter = itertools.count()
        self.km = 0.0
        self._s_last = self.start
        self._dirty: Set[int] = set()
        self._needs_reset = False
        self.expanded = 0
        self.rhs[self.goal] = 0.0
        self._push(self.goal)

    def _calc_key(self, u: int):
        m = min(self.g.get(u, INF), self.rhs.get(u, INF))
        if m == INF:
            return (INF, INF)
        return (m + self.model.h(self.start, u) + self.km, m)

    def _push(self, u: int) -> None:
        k1, k2 = self._calc_key(u)
        tok = next(self._counter)
        self._open[u] = tok
        heapq.heappush(self._heap, (k1, k2, tok, u))

    def _top_key(self):
        while self._heap:
            k1, k2, tok, u = self._heap[0]
            if self._open.get(u) == tok:
                return (k1, k2)
            heapq.heappop(self._heap)             # stale entry (lazy deletion)
        return (INF, INF)

    def _update_vertex(self, u: int) -> None:
        m = self.model
        if u != self.goal:
            best = INF
            for t in m.succ[u].values():
                c = m.edge_cost(t)
                if c < INF:
                    v = c + self.g.get(t.dst, INF)
                    if v < best:
                        best = v
            self.rhs[u] = best
        self._open.pop(u, None)
        if self.g.get(u, INF) != self.rhs.get(u, INF):
            self._push(u)

    def _compute_shortest_path(self) -> None:
        m = self.model
        while True:
            top = self._top_key()
            if top == (INF, INF):
                break
            if not (top < self._calc_key(self.start)
                    or self.rhs.get(self.start, INF) != self.g.get(self.start, INF)):
                break
            k1, k2, tok, u = heapq.heappop(self._heap)
            del self._open[u]
            self.expanded += 1
            k_new = self._calc_key(u)
            if (k1, k2) < k_new:                  # key is outdated because km grew
                self._push(u)
            elif self.g.get(u, INF) > self.rhs.get(u, INF):   # over-consistent
                self.g[u] = self.rhs[u]
                for t in m.pred[u].values():
                    self._update_vertex(t.src)
            else:                                 # under-consistent
                self.g[u] = INF
                self._update_vertex(u)
                for t in m.pred[u].values():
                    self._update_vertex(t.src)

    def _extract_path(self):
        m = self.model
        u = self.start
        spath, tpath, seen = [u], [], {u}
        while u != self.goal:
            best, bv = None, INF
            for t in m.succ[u].values():
                c = m.edge_cost(t)
                if c < INF:
                    v = c + self.g.get(t.dst, INF)
                    if v < bv:
                        best, bv = t, v
            if best is None or best.dst in seen:
                return None, None
            u = best.dst
            seen.add(u)
            spath.append(u)
            tpath.append(best.id)
        return spath, tpath
