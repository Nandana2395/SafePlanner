"""Baseline: A* from scratch (used for verification and for the replanning comparison)."""
from __future__ import annotations

import heapq
import time
from typing import Optional

from graph import GraphModel, INF, Weights, make_result
from models import Planner, PlanningProblem, PlanningResult


class AStarPlanner(Planner):
    def __init__(self, weights: Optional[Weights] = None):
        self.weights = weights or Weights()

    def plan(self, problem: PlanningProblem) -> PlanningResult:
        model = GraphModel(problem, self.weights)
        return self.plan_on_model(model, problem.initial_state, problem.goal_state)

    @staticmethod
    def plan_on_model(model: GraphModel, start: int, goal: int) -> PlanningResult:
        t0 = time.perf_counter()
        if model.blocked(start) or model.blocked(goal):
            return make_result(model, None, None, 0, time.perf_counter() - t0,
                               "start or goal is a bad state")
        g = {start: 0.0}
        parent = {}
        closed = set()
        heap = [(model.h(start, goal), 0.0, start)]
        expanded = 0
        while heap:
            _, gu, u = heapq.heappop(heap)
            if u in closed:
                continue
            closed.add(u)
            expanded += 1
            if u == goal:
                break
            for t in model.succ[u].values():
                c = model.edge_cost(t)
                if c == INF:
                    continue
                ng = gu + c
                if ng < g.get(t.dst, INF):
                    g[t.dst] = ng
                    parent[t.dst] = t
                    heapq.heappush(heap, (ng + model.h(t.dst, goal), ng, t.dst))
        if goal not in closed:
            return make_result(model, None, None, expanded, time.perf_counter() - t0)
        spath, tpath, u = [goal], [], goal
        while u != start:
            t = parent[u]
            tpath.append(t.id)
            u = t.src
            spath.append(u)
        spath.reverse()
        tpath.reverse()
        return make_result(model, spath, tpath, expanded, time.perf_counter() - t0)
