"""Builders for the six illustrative test cases from the assignment."""
from __future__ import annotations

from models import PlanningProblem, State, Transition


class Builder:
    def __init__(self):
        self.ids, self.states, self.trans = {}, [], []

    def state(self, name, *pos):
        self.ids[name] = len(self.states)
        self.states.append(State(self.ids[name], list(pos)))
        return self

    def edge(self, a, b, cost, safety=1.0, rel=1.0):
        self.trans.append(Transition(len(self.trans), self.ids[a], self.ids[b], cost, safety, rel))
        return self

    def problem(self, start, goal, bad=()):
        return PlanningProblem(self.ids[start], self.ids[goal], [self.ids[b] for b in bad],
                               self.states, self.trans)

    def names(self, path):
        inv = {v: k for k, v in self.ids.items()}
        return [inv[i] for i in path]

    def tid(self, a, b):
        for t in self.trans:
            if t.src == self.ids[a] and t.dst == self.ids[b]:
                return t.id
        raise KeyError((a, b))


def case1():   # basic reachability  S->A->B->G
    b = Builder()
    for n, x in zip("SABG", range(4)):
        b.state(n, float(x), 0.0)
    b.edge("S", "A", 1).edge("A", "B", 1).edge("B", "G", 1)
    return b, b.problem("S", "G")


def case2():   # bad-state avoidance: shorter path goes through bad X
    b = Builder()
    b.state("S", 0, 0).state("A", 1, 1).state("X", 2, 1).state("G", 3, 0)
    b.state("C", 1, -1).state("D", 2, -1)
    b.edge("S", "A", 1).edge("A", "X", 1).edge("X", "G", 1)
    b.edge("S", "C", 1.5).edge("C", "D", 1.5).edge("D", "G", 1.5)
    return b, b.problem("S", "G", bad=["X"])


def case3():   # safety margin: cheap path passes next to bad Z, long path stays far away
    b = Builder()
    b.state("S", 0, 0).state("P", 2, 0.3).state("G", 4, 0).state("Z", 2, 0.4)
    b.state("Q", 1, -3).state("R", 3, -3)
    b.edge("S", "P", 2).edge("P", "G", 2)                              # total 4, clearance 0.1
    b.edge("S", "Q", 2.5).edge("Q", "R", 2).edge("R", "G", 2.5)        # total 7, clearance > 3
    return b, b.problem("S", "G", bad=["Z"])


def case4():   # dynamic transition: (A,G) becomes unavailable
    b = Builder()
    b.state("S", 0, 0).state("A", 1, 0).state("G", 2, 0).state("B", 1, -1)
    b.edge("S", "A", 1).edge("A", "G", 1).edge("S", "B", 1.5).edge("B", "G", 1.5)
    return b, b.problem("S", "G")


def case5():   # goal update: G1 -> G2
    b = Builder()
    b.state("S", 0, 0).state("A", 1, 0).state("G1", 2, 0).state("G2", 1, 1)
    b.edge("S", "A", 1).edge("A", "G1", 1).edge("A", "G2", 1)
    return b, b.problem("S", "G1")


def case6():   # transition addition: shortcut S->G
    b = Builder()
    for n, x in zip("SABG", range(4)):
        b.state(n, float(x), 0.0)
    b.edge("S", "A", 2).edge("A", "B", 2).edge("B", "G", 2)
    return b, b.problem("S", "G")
