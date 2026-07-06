"""End-to-end validation that the middleware now EXECUTES a FK join under DP,
closing the gap between the paper's join forecast (validated offline in
join_validation.py) and the running system.

Until now the parser rejected every JOIN, so the join transfer was demonstrated
only on a standalone harness. This script drives a real orders><lineitem COUNT
through the production path -- parse, sensitivity, budget, Laplace, cache -- and
checks the privacy-relevant invariants:

  1. a 2-table inner FK join COUNT executes (no longer rejected);
  2. its sensitivity is the public FK multiplicity d_max=7, not 1 (the privacy
     unit is the order entity);
  3. an exact repeat is served from cache for eps=0 (post-processing);
  4. the single-table twin has sensitivity 1, so the ONLY thing that changed is
     Delta f -- exactly the paper's mechanism-agnostic claim, now in the system;
  5. the unsafe envelope is rejected: SUM/AVG over a join, OUTER/CROSS joins,
     more than one join, and a join whose FK multiplicity is not declared.

Run: python experiments/live_join.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.config import Config
from dpdb.middleware import DPMiddleware, ExecutionMode

JOIN = ("SELECT COUNT(*) FROM orders JOIN lineitem ON l_orderkey = o_orderkey "
        "WHERE o_orderpriority = '{p}'")
SINGLE = "SELECT COUNT(*) FROM orders WHERE o_orderpriority = '{p}'"
PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    ok = bool(cond)
    passed += ok
    failed += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")


def main():
    cfg = Config.from_yaml(str(Path(__file__).parent.parent / "config.yaml"))

    print("=== Positive path: orders><lineitem COUNT runs live under DP ===")
    mw = DPMiddleware(cfg, mode=ExecutionMode.WORKLOAD_DP)
    p = PRIORITIES[0]
    r = mw.execute(JOIN.format(p=p), epsilon=1.0)
    check("join executes (no parse rejection)", r.error is None, str(r.error))
    check("join sensitivity = d_max = 7", r.sensitivities and r.sensitivities[0].sensitivity == 7.0,
          f"got {[s.sensitivity for s in r.sensitivities]}")
    check("join returns a noisy count", r.rows and r.rows[0][0] > 0, str(r.rows))
    check("fresh release costs eps>0", r.epsilon_used == 1.0, f"eps={r.epsilon_used}")

    r2 = mw.execute(JOIN.format(p=p), epsilon=1.0)
    check("exact repeat is a free cache hit", r2.cache_hit and r2.epsilon_used == 0.0,
          f"hit={r2.cache_hit} eps={r2.epsilon_used}")

    rs = mw.execute(SINGLE.format(p=p), epsilon=1.0)
    check("single-table twin sensitivity = 1", rs.sensitivities[0].sensitivity == 1.0,
          f"got {rs.sensitivities[0].sensitivity}")
    check("join count ~ d_avg x single count (joins fan out)",
          rs.rows[0][0] > 0 and r.rows[0][0] > rs.rows[0][0],
          f"join={r.rows[0][0]} single={rs.rows[0][0]}")

    print("\n=== Budget: only DISTINCT join queries spend; repeats are free ===")
    mw2 = DPMiddleware(cfg, mode=ExecutionMode.WORKLOAD_DP)
    spent_each = []
    # issue the 5 distinct join COUNTs, each twice
    for rep in range(2):
        for pr in PRIORITIES:
            res = mw2.execute(JOIN.format(p=pr), epsilon=1.0)
            spent_each.append(res.epsilon_used)
    fresh = sum(1 for e in spent_each if e > 0)
    hits = sum(1 for e in spent_each if e == 0)
    check("5 distinct joins paid once each, 5 repeats free", fresh == 5 and hits == 5,
          f"fresh={fresh} hits={hits}")

    print("\n=== Negative path: the unsafe envelope is rejected ===")
    bad = {
        "SUM over join rejected (future work)":
            "SELECT SUM(l_quantity) FROM orders JOIN lineitem ON l_orderkey=o_orderkey",
        "AVG over join rejected":
            "SELECT AVG(l_quantity) FROM orders JOIN lineitem ON l_orderkey=o_orderkey",
        "LEFT OUTER join rejected":
            "SELECT COUNT(*) FROM orders LEFT JOIN lineitem ON l_orderkey=o_orderkey",
        "two joins rejected":
            ("SELECT COUNT(*) FROM orders JOIN lineitem ON l_orderkey=o_orderkey "
             "JOIN customer ON o_custkey=c_custkey"),
    }
    for name, sql in bad.items():
        res = mw.execute(sql, epsilon=1.0)
        check(name, res.error is not None and (res.rows == [] or not res.rows),
              str(res.error))

    # a join whose FK multiplicity is NOT declared must be refused, not guessed
    cfg_nojoin = Config.from_yaml(str(Path(__file__).parent.parent / "config.yaml"))
    cfg_nojoin.fk_multiplicity = {}
    mw3 = DPMiddleware(cfg_nojoin, mode=ExecutionMode.WORKLOAD_DP)
    res = mw3.execute(JOIN.format(p=p), epsilon=1.0)
    check("undeclared FK multiplicity rejected (no unbounded-sensitivity guess)",
          res.error is not None and "d_max" in str(res.error), str(res.error))

    print(f"\n=== {passed} passed, {failed} failed ===")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
