"""Experimental evaluation: D* Lite replanning vs A* from scratch on random geometric graphs.

Run:  python experiments.py            -> prints a table, writes results.csv (+ results.png)
"""
from __future__ import annotations

import csv
import time
import tracemalloc

import numpy as np

from astar import AStarPlanner
from dstar_lite import DStarLitePlanner
from models import PlanningProblem, State, Transition


def make_random_problem(n: int, k: int = 8, bad_frac: float = 0.05, seed: int = 0) -> PlanningProblem:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 100, (n, 2))
    states = [State(i, X[i].tolist()) for i in range(n)]
    edges = set()
    for lo in range(0, n, 500):                           # chunked k-nearest-neighbours
        D = np.linalg.norm(X[lo:lo + 500, None, :] - X[None, :, :], axis=2)
        for r in range(D.shape[0]):
            i = lo + r
            for j in np.argpartition(D[r], k)[:k + 1]:
                if int(j) != i:
                    edges.add((i, int(j)))
                    edges.add((int(j), i))
    transitions = []
    for tid, (a, b) in enumerate(sorted(edges)):
        d = float(np.linalg.norm(X[a] - X[b]))
        transitions.append(Transition(tid, a, b, d * (1 + rng.uniform(0, 0.5)),
                                      safety=float(rng.uniform(0.5, 1)),
                                      reliability=float(rng.uniform(0.7, 1))))
    start = int(np.argmin(np.linalg.norm(X - [0, 0], axis=1)))
    goal = int(np.argmin(np.linalg.norm(X - [100, 100], axis=1)))
    pool = [i for i in range(n) if i not in (start, goal)]
    bad = rng.choice(pool, size=int(bad_frac * n), replace=False).tolist()
    return PlanningProblem(start, goal, bad, states, transitions)


def run(n: int, events: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed + 1)
    prob = make_random_problem(n, seed=seed)
    rows = []

    # ---- initial planning: timing pass (no tracemalloc) ----
    t0 = time.perf_counter()
    dpl = DStarLitePlanner()
    r = dpl.plan(prob)
    d_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    ra = AStarPlanner().plan(prob)
    a_time = time.perf_counter() - t0
    # ---- memory pass ----
    tracemalloc.start()
    DStarLitePlanner().plan(prob)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rows.append(dict(n=n, event="initial", dstar_ms=d_time * 1e3, dstar_expanded=r.expanded,
                     astar_ms=a_time * 1e3, astar_expanded=ra.expanded,
                     cost=r.total_cost, min_dist_bad=r.safety_score, bad_visited=r.bad_states_visited,
                     success=r.success, costs_match=abs(r.effective_cost - ra.effective_cost) < 1e-6,
                     peak_mem_mb=peak / 1e6))

    kinds = ["block_transitions", "new_bad_states", "add_shortcut", "goal_change"]
    next_tid = max(t.id for t in prob.transitions) + 1
    for e in range(events):
        kind = kinds[e % len(kinds)]
        path = r.state_path if r.success else [dpl.start]
        tpath = r.transition_path if r.success else []
        if kind == "block_transitions" and tpath:
            for tid in rng.choice(tpath, size=min(3, len(tpath)), replace=False):
                dpl.set_transition_available(int(tid), False)
        elif kind == "new_bad_states" and len(path) > 2:
            inner = path[1:-1]
            new = [int(rng.choice(inner))] + [int(x) for x in rng.integers(0, n, 3)]
            new = [x for x in new if x not in (dpl.start, dpl.goal)]
            dpl.set_bad_states(set(dpl.model.bad) | set(new))
        elif kind == "add_shortcut" and len(path) > 4:
            a, b = path[0], path[(2 * len(path)) // 3]
            d = float(np.linalg.norm(dpl.model.states[a] - dpl.model.states[b]))
            dpl.add_transition(Transition(next_tid, a, b, d))
            next_tid += 1
        elif kind == "goal_change":
            cand = [int(x) for x in rng.integers(0, n, 20) if not dpl.model.blocked(int(x))]
            if cand:
                dpl.set_goal(cand[0])
        t0 = time.perf_counter()
        r = dpl.replan()
        d_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        ra = AStarPlanner.plan_on_model(dpl.model, dpl.start, dpl.goal)
        a_time = time.perf_counter() - t0
        match = (r.success == ra.success) and (not r.success or abs(r.effective_cost - ra.effective_cost) < 1e-6)
        rows.append(dict(n=n, event=kind, dstar_ms=d_time * 1e3, dstar_expanded=r.expanded,
                         astar_ms=a_time * 1e3, astar_expanded=ra.expanded,
                         cost=r.total_cost, min_dist_bad=r.safety_score, bad_visited=r.bad_states_visited,
                         success=r.success, costs_match=match, peak_mem_mb=peak / 1e6))
    return rows


def main():
    all_rows = []
    for n in (500, 2000, 5000):
        all_rows += run(n)
    keys = list(all_rows[0].keys())
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    print(f"{'n':>5} {'event':<18}{'D* ms':>9}{'D* exp':>8}{'A* ms':>9}{'A* exp':>8}"
          f"{'cost':>9}{'minDist':>9}{'bad':>4}{'ok':>4}{'match':>6}")
    for r in all_rows:
        print(f"{r['n']:>5} {r['event']:<18}{r['dstar_ms']:>9.2f}{r['dstar_expanded']:>8}"
              f"{r['astar_ms']:>9.2f}{r['astar_expanded']:>8}{r['cost']:>9.1f}"
              f"{r['min_dist_bad']:>9.2f}{r['bad_visited']:>4}{int(r['success']):>4}{int(r['costs_match']):>6}")
    print("\nPeak memory (MB) of initial D* planning:",
          {r['n']: round(r['peak_mem_mb'], 1) for r in all_rows if r['event'] == 'initial'})
    print("All replans optimal (D* == A*):", all(r['costs_match'] for r in all_rows))
    print("Bad states visited in total   :", sum(r['bad_visited'] for r in all_rows))

    try:                                                  # optional plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for n in (500, 2000, 5000):
            rs = [r for r in all_rows if r['n'] == n and r['event'] != 'initial']
            ax[0].plot([r['dstar_expanded'] for r in rs], 'o-', label=f"D* n={n}")
            ax[0].plot([r['astar_expanded'] for r in rs], 'x--', label=f"A* n={n}")
        ax[0].set_yscale('log'); ax[0].set_xlabel('update #'); ax[0].set_ylabel('states expanded')
        ax[0].legend(fontsize=7); ax[0].set_title('Replanning effort')
        ns = (500, 2000, 5000)
        dm = [np.mean([r['dstar_ms'] for r in all_rows if r['n'] == n and r['event'] != 'initial']) for n in ns]
        am = [np.mean([r['astar_ms'] for r in all_rows if r['n'] == n and r['event'] != 'initial']) for n in ns]
        xs = np.arange(len(ns))
        ax[1].bar(xs - 0.2, dm, 0.4, label='D* Lite'); ax[1].bar(xs + 0.2, am, 0.4, label='A* scratch')
        ax[1].set_xticks(xs); ax[1].set_xticklabels(ns); ax[1].set_xlabel('number of states')
        ax[1].set_ylabel('mean replanning time (ms)'); ax[1].legend()
        plt.tight_layout(); plt.savefig("results.png", dpi=120)
        print("saved results.csv and results.png")
    except ImportError:
        print("saved results.csv (install matplotlib for the plot)")


if __name__ == "__main__":
    main()
