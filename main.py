"""Demo: runs the six illustrative test cases and prints the results."""
from dstar_lite import DStarLitePlanner
from astar import AStarPlanner
from graph import Weights
from models import Transition, score
import scenarios as sc


def show(title, b, r):
    print(f"\n=== {title} ===")
    if not r.success:
        print("  FAILED:", r.message)
        return
    print("  path           :", " -> ".join(b.names(r.state_path)))
    print(f"  total cost     : {r.total_cost:.3f}   (effective {r.effective_cost:.3f})")
    print(f"  min dist to bad: {r.safety_score:.3f}   bad visited: {r.bad_states_visited}")
    print(f"  expanded       : {r.expanded}   time: {r.planning_time * 1e3:.3f} ms   score: {score(r):.2f}")


def main():
    b, p = sc.case1(); show("Test 1: basic reachability", b, DStarLitePlanner().plan(p))
    b, p = sc.case2(); show("Test 2: bad-state avoidance", b, DStarLitePlanner().plan(p))

    b, p = sc.case3()
    show("Test 3a: safety-aware (w_proximity=5)", b, DStarLitePlanner().plan(p))
    show("Test 3b: cost only (w_proximity=0)  ", b, DStarLitePlanner(Weights(w_proximity=0)).plan(p))

    b, p = sc.case4(); pl = DStarLitePlanner()
    show("Test 4: before update", b, pl.plan(p))
    pl.set_transition_available(b.tid("A", "G"), False)
    show("Test 4: after (A,G) removed", b, pl.replan())

    b, p = sc.case5(); pl = DStarLitePlanner()
    show("Test 5: goal = G1", b, pl.plan(p))
    pl.set_goal(b.ids["G2"])
    show("Test 5: goal changed to G2", b, pl.replan())

    b, p = sc.case6(); pl = DStarLitePlanner()
    show("Test 6: before shortcut", b, pl.plan(p))
    pl.add_transition(Transition(99, b.ids["S"], b.ids["G"], 3.0))
    show("Test 6: after shortcut S->G", b, pl.replan())


if __name__ == "__main__":
    main()
