import unittest

import scenarios as sc
from astar import AStarPlanner
from dstar_lite import DStarLitePlanner
from graph import Weights
from models import Transition


class TestPlanner(unittest.TestCase):
    def test1_reachability(self):
        b, p = sc.case1()
        r = DStarLitePlanner().plan(p)
        self.assertTrue(r.success)
        self.assertEqual(b.names(r.state_path), ["S", "A", "B", "G"])

    def test2_bad_state_avoidance(self):
        b, p = sc.case2()
        r = DStarLitePlanner().plan(p)
        self.assertEqual(b.names(r.state_path), ["S", "C", "D", "G"])
        self.assertEqual(r.bad_states_visited, 0)

    def test3_safety_margin(self):
        b, p = sc.case3()
        safe = DStarLitePlanner().plan(p)
        fast = DStarLitePlanner(Weights(w_proximity=0)).plan(p)
        self.assertEqual(b.names(safe.state_path), ["S", "Q", "R", "G"])
        self.assertEqual(b.names(fast.state_path), ["S", "P", "G"])
        self.assertGreater(safe.safety_score, fast.safety_score)
        self.assertLess(fast.total_cost, safe.total_cost)

    def test4_dynamic_transition(self):
        b, p = sc.case4()
        pl = DStarLitePlanner()
        self.assertEqual(b.names(pl.plan(p).state_path), ["S", "A", "G"])
        pl.set_transition_available(b.tid("A", "G"), False)
        self.assertEqual(b.names(pl.replan().state_path), ["S", "B", "G"])

    def test5_goal_update(self):
        b, p = sc.case5()
        pl = DStarLitePlanner()
        pl.plan(p)
        pl.set_goal(b.ids["G2"])
        self.assertEqual(b.names(pl.replan().state_path), ["S", "A", "G2"])

    def test6_transition_addition(self):
        b, p = sc.case6()
        pl = DStarLitePlanner()
        self.assertEqual(pl.plan(p).total_cost, 6)
        pl.add_transition(Transition(99, b.ids["S"], b.ids["G"], 3.0))
        r = pl.replan()
        self.assertEqual(b.names(r.state_path), ["S", "G"])
        self.assertEqual(r.total_cost, 3.0)

    def test_unreachable_when_goal_is_bad(self):
        b, p = sc.case1()
        p.bad_states = [b.ids["G"]]
        self.assertFalse(DStarLitePlanner().plan(p).success)

    def test_dstar_matches_astar_after_updates(self):
        b, p = sc.case4()
        pl = DStarLitePlanner()
        pl.plan(p)
        pl.set_transition_available(b.tid("A", "G"), False)
        r1 = pl.replan()
        r2 = AStarPlanner.plan_on_model(pl.model, pl.start, pl.goal)
        self.assertAlmostEqual(r1.effective_cost, r2.effective_cost)


if __name__ == "__main__":
    unittest.main()
