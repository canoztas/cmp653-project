"""Running-system PERFORMANCE of the live DP middleware.

The paper argues the workload-aware DP layer is a PRACTICAL middleware, not just
an offline accounting story. This script measures what the running system
actually costs, end to end, over the real Adult and TPC-H tables through the
exact production path (parse -> sensitivity -> budget -> Laplace -> cache):

  (a) per-query end-to-end LATENCY, split by path:
        - cache-MISS  : full path, hits DuckDB and adds calibrated noise;
        - cache-HIT   : exact repeat, served from the cache for eps=0
                        (short-circuits before sensitivity/budget);
        - raw DuckDB  : the same SQL run directly on the connection, with NO
                        DP layer -- the floor the middleware is measured against.
  (b) THROUGHPUT (queries/sec) on a deterministic repeated workload that mixes
      fresh releases and cache hits, the regime the middleware is built for.
  (c) the workload-aware middleware OVERHEAD vs raw DuckDB: how many ms / what
      multiple the DP layer adds on the MISS path, and how cheap the HIT path is.

Everything is timed with time.perf_counter over many repetitions; we report the
MEDIAN, p95, and a 95% bootstrap CI on the median so the numbers are defensible.
The system is touched READ-ONLY: we only time mw.execute(...) and raw
con.execute(...); no src/ file is modified and no value depends on wall-clock
time or unseeded randomness (Laplace noise uses the middleware's own RNG, which
does not affect the timing we report).

Run: python experiments/system_perf.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from dpdb.config import Config
from dpdb.middleware import DPMiddleware, ExecutionMode

ROOT = Path(__file__).parent.parent

# Deterministic query bank over the two real datasets. Each is a single-table
# aggregate the middleware fully supports; the {v} slot lets us mint distinct
# fresh species for the MISS / throughput measurements without changing shape.
ADULT_COUNT = "SELECT COUNT(*) FROM adult WHERE age > {v}"
ADULT_SUM = "SELECT SUM(hours_per_week) FROM adult WHERE age > {v}"
ADULT_GROUP = "SELECT sex, COUNT(*) FROM adult WHERE age > {v} GROUP BY sex"
TPCH_COUNT = "SELECT COUNT(*) FROM orders WHERE o_totalprice > {v}"
TPCH_SUM = "SELECT SUM(l_quantity) FROM lineitem WHERE l_quantity > {v}"

# A fixed value per shape that selects a non-trivial slice (used for HIT timing
# and as the raw baseline). Values are constants, not random.
SHAPES = {
    "adult_count": (ADULT_COUNT, 30),
    "adult_sum": (ADULT_SUM, 30),
    "adult_group": (ADULT_GROUP, 30),
    "tpch_count": (TPCH_COUNT, 100000),
    "tpch_sum": (TPCH_SUM, 10),
}

# How many timed repetitions per cell. Latencies here are sub-millisecond to a
# few ms, so we need many reps for a stable median / tight CI.
REPS_HIT = 400
REPS_MISS = 200        # raw baseline uses this same count (matched SQL)
WARMUP = 20            # discard the first few (import/JIT/page-cache warmup)
BOOT = 2000            # bootstrap resamples for the median CI
BOOT_SEED = 20260626


def _median_ci(samples: np.ndarray, n_boot: int = BOOT, seed: int = BOOT_SEED):
    """95% bootstrap CI on the median. Deterministic (seeded RNG)."""
    rng = np.random.default_rng(seed)
    n = len(samples)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_meds = np.median(samples[idx], axis=1)
    lo, hi = np.percentile(boot_meds, [2.5, 97.5])
    return float(lo), float(hi)


def _summ(name: str, path: str, samples_ms: np.ndarray) -> dict:
    s = np.asarray(samples_ms, dtype=float)
    lo, hi = _median_ci(s)
    return dict(
        cell=name,
        path=path,
        n=len(s),
        median_ms=float(np.median(s)),
        median_ci_lo_ms=lo,
        median_ci_hi_ms=hi,
        p95_ms=float(np.percentile(s, 95)),
        mean_ms=float(np.mean(s)),
        min_ms=float(np.min(s)),
    )


def _fresh_mw(cfg: Config, big_budget: bool = False) -> DPMiddleware:
    """A WORKLOAD_DP middleware. With big_budget, raise total_epsilon so a long
    run of DISTINCT fresh releases never exhausts the ledger mid-measurement
    (we are timing the system path, not exercising the budget cap here)."""
    if big_budget:
        cfg = Config.from_yaml(str(ROOT / "config.yaml"))
        cfg.privacy.total_epsilon = 1e12
    return DPMiddleware(cfg, mode=ExecutionMode.WORKLOAD_DP)


def _varied_sql(template: str, base_v: int, reps: int) -> list[str]:
    """The exact list of distinct-literal queries used for BOTH the raw and the
    MISS measurements, so the DP-overhead comparison is apples-to-apples (the two
    paths run identical SQL with identical selectivity, differing ONLY in the DP
    layer)."""
    return [template.format(v=base_v + i) for i in range(reps)]


def time_raw(mw: DPMiddleware, warm: list[str], timed: list[str]) -> np.ndarray:
    """Time each SQL in `timed` directly on the DuckDB connection -- no DP layer.
    This is the floor: parse, sensitivity, budget, noise, cache are all absent.
    We time the SAME varied-literal queries the MISS path runs, so subtracting
    the two medians isolates exactly the cost the DP layer adds. `warm` is run
    (untimed) first to settle the page cache."""
    con = mw.db.conn
    for s in warm:
        con.execute(s).fetchall()
    out = np.empty(len(timed))
    for i, s in enumerate(timed):
        t0 = time.perf_counter()
        con.execute(s).fetchall()
        out[i] = (time.perf_counter() - t0) * 1000.0
    return out


def time_hit(cfg: Config, sql: str, reps: int) -> np.ndarray:
    """Time the cache-HIT path: release once (fresh), then repeat the EXACT query
    many times -- each repeat is served from cache at eps=0. We time the repeats
    only. Uses mw.execute's own latency_ms so we measure exactly the path the
    system reports to a client."""
    mw = _fresh_mw(cfg, big_budget=True)
    first = mw.execute(sql, epsilon=1.0)
    assert first.error is None, first.error
    assert not first.cache_hit, "first release must be a MISS"
    for _ in range(WARMUP):
        mw.execute(sql, epsilon=1.0)
    out = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter()
        r = mw.execute(sql, epsilon=1.0)
        out[i] = (time.perf_counter() - t0) * 1000.0
        assert r.cache_hit and r.epsilon_used == 0.0, "expected a free cache hit"
    return out


def time_miss(cfg: Config, warm: list[str], timed: list[str]) -> np.ndarray:
    """Time the cache-MISS path: every call is a DISTINCT fresh species (the
    WHERE literal varies), so it always traverses the full path and hits DuckDB.
    Runs the SAME `timed` SQL list as time_raw. `warm` is a DISJOINT set of
    literals run (untimed) first so the timed calls are still genuine MISSes.
    Big budget so the ledger never exhausts."""
    mw = _fresh_mw(cfg, big_budget=True)
    for s in warm:
        mw.execute(s, epsilon=1.0)
    out = np.empty(len(timed))
    for i, s in enumerate(timed):
        t0 = time.perf_counter()
        r = mw.execute(s, epsilon=1.0)
        out[i] = (time.perf_counter() - t0) * 1000.0
        assert r.error is None, r.error
        assert not r.cache_hit, "varied literal must be a cache MISS"
    return out


def measure_throughput(cfg: Config, seed: int = 7) -> dict:
    """Throughput (q/s) on a deterministic mixed repeated workload: a small pool
    of distinct species drawn with a skew that produces many exact repeats
    (cache hits) interleaved with fresh releases -- the workload the middleware
    is designed for. Deterministic draw (seeded). Big budget so the run is not
    cut short by exhaustion; we are measuring served throughput, not the cap."""
    mw = _fresh_mw(cfg, big_budget=True)
    rng = np.random.default_rng(seed)

    # A realistic mixed pool: many distinct species across both datasets, drawn
    # with a moderate Zipf skew so the stream has BOTH repeats (cache hits, the
    # regime the middleware exploits) AND a steady stream of fresh releases
    # (MISSes, the full path). With ~150 species and 3000 queries the realized
    # hit rate lands well below 100%, so throughput reflects the true mix, not
    # the hit path alone.
    pool = []
    for v in range(300):
        pool.append(ADULT_COUNT.format(v=20 + v))
        pool.append(ADULT_SUM.format(v=20 + v))
        pool.append(TPCH_COUNT.format(v=50000 + 200 * v))
        pool.append(TPCH_SUM.format(v=2 + (v % 45)))
        pool.append(ADULT_GROUP.format(v=20 + v))
    ranks = np.arange(1, len(pool) + 1)
    p = (1.0 / ranks ** 0.8)
    p = p / p.sum()

    n_queries = 3000
    order = rng.choice(len(pool), size=n_queries, p=p)

    # warmup
    for j in range(WARMUP):
        mw.execute(pool[order[j % len(order)]], epsilon=1.0)

    hits = 0
    t0 = time.perf_counter()
    for idx in order:
        r = mw.execute(pool[idx], epsilon=1.0)
        assert r.error is None, r.error
        hits += int(r.cache_hit)
    elapsed = time.perf_counter() - t0

    qps = n_queries / elapsed
    return dict(
        cell="mixed_workload",
        path="throughput",
        n_queries=n_queries,
        distinct_species=len(pool),
        cache_hit_rate=hits / n_queries,
        elapsed_s=elapsed,
        throughput_qps=qps,
        mean_ms_per_query=elapsed / n_queries * 1000.0,
    )


def main():
    cfg = Config.from_yaml(str(ROOT / "config.yaml"))

    print("=== System performance of the live DP middleware (real Adult + TPC-H) ===")
    print(f"    reps: miss=raw={REPS_MISS} hit={REPS_HIT}, "
          f"warmup={WARMUP}, median CI via {BOOT} bootstraps\n")

    rows = []
    print(f"  {'cell':>12} | {'path':>5} | {'median':>8} | "
          f"{'95% CI (median)':>18} | {'p95':>8}")
    print("  " + "-" * 66)

    for name, (template, v) in SHAPES.items():
        fixed_sql = template.format(v=v)

        # Disjoint warmup vs timed literals so timed MISS calls are genuine
        # misses; raw and miss are timed on the IDENTICAL `timed` SQL list.
        warm = _varied_sql(template, v - 5000, WARMUP)
        timed = _varied_sql(template, v, REPS_MISS)

        raw = time_raw(_fresh_mw(cfg), warm, timed)
        hit = time_hit(cfg, fixed_sql, REPS_HIT)
        miss = time_miss(cfg, warm, timed)

        for path, samples in (("raw", raw), ("hit", hit), ("miss", miss)):
            s = _summ(name, path, samples)
            rows.append(s)
            print(f"  {name:>12} | {path:>5} | {s['median_ms']:7.3f}ms | "
                  f"[{s['median_ci_lo_ms']:6.3f},{s['median_ci_hi_ms']:6.3f}] | "
                  f"{s['p95_ms']:6.3f}ms")

    df = pd.DataFrame(rows)

    # Overhead of the DP layer vs raw DuckDB, per shape (MISS path is the fair
    # comparison: both hit the DB; HIT path skips the DB entirely).
    piv = df.pivot_table(index="cell", columns="path", values="median_ms")
    piv["miss_over_raw_ms"] = piv["miss"] - piv["raw"]
    piv["miss_over_raw_x"] = piv["miss"] / piv["raw"]
    piv["hit_speedup_vs_miss_x"] = piv["miss"] / piv["hit"]
    piv = piv.reset_index()

    print("\n=== DP-layer overhead vs raw DuckDB (per shape, medians) ===")
    print(f"  {'cell':>12} | {'raw':>8} | {'miss':>8} | {'+DP':>8} | "
          f"{'miss/raw':>8} | {'hit':>8} | {'miss/hit':>8}")
    print("  " + "-" * 78)
    for _, r in piv.iterrows():
        print(f"  {r['cell']:>12} | {r['raw']:7.3f}ms | {r['miss']:7.3f}ms | "
              f"{r['miss_over_raw_ms']:+7.3f}ms | {r['miss_over_raw_x']:7.2f}x | "
              f"{r['hit']:7.3f}ms | {r['hit_speedup_vs_miss_x']:7.2f}x")

    tput = measure_throughput(cfg)
    print("\n=== Throughput on a deterministic mixed repeated workload ===")
    print(f"  {tput['n_queries']} queries over {tput['distinct_species']} distinct "
          f"species, cache-hit rate {tput['cache_hit_rate']*100:.1f}%")
    print(f"  served in {tput['elapsed_s']:.3f}s -> "
          f"{tput['throughput_qps']:.0f} q/s "
          f"({tput['mean_ms_per_query']:.3f} ms/query mean)")

    # Persist everything: per-cell latency rows + the throughput row, in one CSV.
    out_dir = ROOT / "results" / "perf"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "system_perf.csv"

    tput_row = {
        "cell": tput["cell"], "path": tput["path"],
        "n": tput["n_queries"],
        "median_ms": tput["mean_ms_per_query"],
        "median_ci_lo_ms": float("nan"), "median_ci_hi_ms": float("nan"),
        "p95_ms": float("nan"), "mean_ms": tput["mean_ms_per_query"],
        "min_ms": float("nan"),
        "throughput_qps": tput["throughput_qps"],
        "cache_hit_rate": tput["cache_hit_rate"],
        "distinct_species": tput["distinct_species"],
        "elapsed_s": tput["elapsed_s"],
    }
    full = pd.concat([df, pd.DataFrame([tput_row])], ignore_index=True)
    full.to_csv(out_path, index=False)

    # Headline numbers, aggregated across shapes.
    miss_med = df[df.path == "miss"].median_ms
    raw_med = df[df.path == "raw"].median_ms
    hit_med = df[df.path == "hit"].median_ms
    over_ms = piv["miss_over_raw_ms"]
    over_x = piv["miss_over_raw_x"]

    print("\n=== Headline ===")
    print(f"  Cache-MISS (full DP path): median {miss_med.min():.3f}-"
          f"{miss_med.max():.3f} ms across {len(SHAPES)} query shapes.")
    print(f"  Cache-HIT  (free repeat) : median {hit_med.min():.3f}-"
          f"{hit_med.max():.3f} ms -- the eps=0 reuse path.")
    print(f"  Raw DuckDB (no DP layer) : median {raw_med.min():.3f}-"
          f"{raw_med.max():.3f} ms.")
    print(f"  DP-layer overhead on the MISS path: +{over_ms.min():.3f} to "
          f"+{over_ms.max():.3f} ms ({over_x.min():.2f}x-{over_x.max():.2f}x raw).")
    print(f"  A cache HIT is {(miss_med.median()/hit_med.median()):.1f}x faster "
          f"than recomputing the noisy answer (and free in eps).")
    print(f"  Mixed repeated workload throughput: "
          f"{tput['throughput_qps']:.0f} q/s at "
          f"{tput['cache_hit_rate']*100:.0f}% hit rate.")
    print(f"\n  Wrote {out_path} ({len(full)} rows).")


# --------------------------------------------------------------------------- #
# Self-tests (pytest-discoverable). Run: pytest experiments/system_perf.py
# These validate the MEASUREMENT HARNESS itself: that the paths it times are the
# paths it claims (HIT really hits, MISS really misses), the raw/miss SQL is
# identical, the stats are correct, and the CSV/headline numbers are produced.
# They are deliberately small (few reps) so they run in well under a second.
# --------------------------------------------------------------------------- #

def _cfg():
    return Config.from_yaml(str(ROOT / "config.yaml"))


def test_median_ci_basic():
    s = np.arange(1.0, 101.0)
    lo, hi = _median_ci(s, n_boot=500, seed=1)
    assert lo <= np.median(s) <= hi
    # deterministic: same seed -> identical CI
    assert _median_ci(s, n_boot=500, seed=1) == _median_ci(s, n_boot=500, seed=1)


def test_summ_shape():
    s = _summ("c", "raw", np.array([1.0, 2.0, 3.0, 4.0]))
    assert s["n"] == 4 and s["median_ms"] == 2.5
    assert s["median_ci_lo_ms"] <= s["median_ms"] <= s["median_ci_hi_ms"]
    assert s["p95_ms"] >= s["median_ms"] >= s["min_ms"]


def test_varied_sql_distinct():
    sqls = _varied_sql(ADULT_COUNT, 30, 8)
    assert len(sqls) == 8 and len(set(sqls)) == 8  # all distinct literals


def test_hit_path_is_a_real_cache_hit():
    """time_hit must traverse the eps=0 cache path: first call MISS, repeats HIT."""
    cfg = _cfg()
    mw = _fresh_mw(cfg, big_budget=True)
    sql = ADULT_COUNT.format(v=40)
    first = mw.execute(sql, epsilon=1.0)
    assert first.error is None and not first.cache_hit and first.epsilon_used == 1.0
    rep = mw.execute(sql, epsilon=1.0)
    assert rep.cache_hit and rep.epsilon_used == 0.0
    # and the timing wrapper returns the right count of positive samples
    out = time_hit(cfg, sql, 5)
    assert len(out) == 5 and np.all(out >= 0)


def test_miss_path_is_a_real_cache_miss_and_matches_raw_sql():
    """time_miss must never serve a cache hit; raw and miss run identical SQL."""
    cfg = _cfg()
    warm = _varied_sql(TPCH_COUNT, 10000, 3)
    timed = _varied_sql(TPCH_COUNT, 100000, 6)
    assert not (set(warm) & set(timed))  # disjoint warmup vs timed
    miss = time_miss(cfg, warm, timed)
    raw = time_raw(_fresh_mw(cfg), warm, timed)
    assert len(miss) == 6 and len(raw) == 6
    # On real data the full DP path is strictly slower than raw DuckDB.
    assert np.median(miss) > np.median(raw)


def test_throughput_mixes_both_paths():
    """The throughput workload must exercise BOTH paths (some hits, some misses),
    not collapse to a single path, and report a positive q/s."""
    t = measure_throughput(_cfg(), seed=3)
    assert t["throughput_qps"] > 0
    assert 0.0 < t["cache_hit_rate"] < 1.0  # genuine mix, not all-hit / all-miss
    assert t["distinct_species"] > 100


if __name__ == "__main__":
    main()
