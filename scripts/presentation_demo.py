"""Scripted demo for the project presentation.

Runs through 4 scenarios that demonstrate the key contributions of the
DP-SQL middleware. Each scenario pauses for narration.

Run:
    python scripts/presentation_demo.py
    python scripts/presentation_demo.py --auto      # no pauses
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box

from dpdb.budget import AllocationStrategy, BudgetLedger
from dpdb.config import Config
from dpdb.demo import compare_mode, trace_query
from dpdb.middleware import DPMiddleware, ExecutionMode


console = Console(force_terminal=True, legacy_windows=False)


def title(text: str):
    console.print()
    console.print(Panel(
        f"[bold white]{text}[/bold white]",
        border_style="bold magenta", box=box.DOUBLE_EDGE))
    console.print()


def narration(text: str):
    console.print(Panel(text, border_style="cyan", title="[bold]Narration[/bold]"))


def pause(auto: bool):
    if not auto:
        console.print("\n[dim]>> Press Enter to continue...[/dim]", end="")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
    else:
        time.sleep(2)


def scenario_1(config, auto: bool):
    title("SCENARIO 1: Step-by-step trace of a single private query")
    narration(
        "We run a COUNT query over the UCI Adult dataset (48,842 rows).\n"
        "The middleware will show 9 steps: parse, template extraction,\n"
        "cache check, sensitivity analysis, budget allocation,\n"
        "DB execution, noise injection, and cache store."
    )
    pause(auto)
    ledger = BudgetLedger(config.privacy.total_epsilon, AllocationStrategy.WORKLOAD_AWARE)
    trace_query("SELECT COUNT(*) FROM adult WHERE age > 30", config,
                epsilon=1.0, budget_ledger=ledger)
    pause(auto)


def scenario_2(config, auto: bool):
    title("SCENARIO 2: Same query repeated -> CACHE HIT (eps = 0)")
    narration(
        "Now the analyst issues the EXACT SAME query again.\n"
        "Workload-aware mode detects the match by template+parameter hash\n"
        "and returns the cached noisy answer at ZERO additional privacy cost.\n"
        "This is justified by the post-processing property of DP."
    )
    pause(auto)
    ledger = BudgetLedger(config.privacy.total_epsilon, AllocationStrategy.WORKLOAD_AWARE)
    sql = "SELECT COUNT(*) FROM adult WHERE age > 30"
    console.print("\n[bold]>> First call:[/bold] (cache miss)")
    trace_query(sql, config, epsilon=1.0, budget_ledger=ledger)
    pause(auto)
    console.print("\n[bold green]>> Second call:[/bold green] (same query!)")
    trace_query(sql, config, epsilon=1.0, budget_ledger=ledger)
    pause(auto)


def scenario_3(config, auto: bool):
    title("SCENARIO 3: Baseline Comparison (Exact vs PINQ vs Ours)")
    narration(
        "We run the same query under three strategies:\n"
        " * Exact (no privacy) -- gold standard, baseline truth\n"
        " * PINQ / Naive DP    -- standard textbook DP-SQL approach\n"
        " * Workload-Aware     -- our contribution\n\n"
        "When the query is REPEATED, only ours saves budget."
    )
    pause(auto)
    compare_mode("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'",
                 config, epsilon=1.0)
    pause(auto)


def scenario_noise_visible(config, auto: bool):
    """Show noise actually changing answers and the caching difference."""
    title("SCENARIO: Noise IS real -- 5 repeated runs, Naive vs Workload-Aware")
    narration(
        "We pick a query that returns a SMALL count and use a SMALL epsilon\n"
        "so the Laplace noise is clearly visible.\n\n"
        "Query: COUNT of people aged 90+ in the Adult dataset (true value = 55)\n"
        "Epsilon: 0.1 (so noise scale = 1/0.1 = 10 -- noise visible!)\n\n"
        "We will run the SAME query 5 times:\n"
        " * Naive DP:    adds FRESH noise each time -> 5 different answers,\n"
        "                consumes 5 x 0.1 = 0.5 eps total.\n"
        " * Workload-DP: first call adds noise, the 4 follow-ups return the\n"
        "                IDENTICAL cached answer -> 1 noisy value, 0.1 eps total.\n"
    )
    pause(auto)

    sql = "SELECT COUNT(*) FROM adult WHERE age >= 90"
    eps = 0.1

    # True answer
    mw_exact = DPMiddleware(config, mode=ExecutionMode.EXACT)
    true_val = mw_exact.execute(sql).rows[0][0]

    console.print(f"\n[bold]Query:[/bold] [cyan]{sql}[/cyan]")
    console.print(f"[bold]True answer (exact SQL):[/bold] [green]{true_val}[/green]")
    console.print(f"[bold]Noise scale:[/bold] Lap(b = 1/{eps}) = [yellow]Lap(10)[/yellow]\n")

    # Naive: 5 runs, each adds fresh noise
    np_seed = 42
    import numpy as np
    np.random.seed(np_seed)
    mw_naive = DPMiddleware(config, mode=ExecutionMode.NAIVE_DP)
    naive_runs = []
    for i in range(5):
        r = mw_naive.execute(sql, epsilon=eps)
        naive_runs.append(r.rows[0][0])

    np.random.seed(np_seed)
    mw_wa = DPMiddleware(config, mode=ExecutionMode.WORKLOAD_DP)
    wa_runs = []
    wa_hits = []
    for i in range(5):
        r = mw_wa.execute(sql, epsilon=eps)
        wa_runs.append(r.rows[0][0])
        wa_hits.append(r.cache_hit)

    from rich.table import Table
    t = Table(title="Same query, 5 times -- noise visible, caching deterministic",
              box=box.DOUBLE_EDGE)
    t.add_column("Run", justify="center", style="bold")
    t.add_column("True answer", justify="right", style="green")
    t.add_column("Naive DP answer", justify="right", style="red")
    t.add_column("|err|", justify="right", style="red")
    t.add_column("Workload-Aware answer", justify="right", style="bold green")
    t.add_column("Cache hit?", justify="center")

    for i in range(5):
        err_naive = abs(naive_runs[i] - true_val)
        cache_mark = "[bold green]YES (eps=0)[/bold green]" if wa_hits[i] else "[yellow]NO[/yellow]"
        t.add_row(
            f"#{i+1}",
            f"{true_val}",
            f"{naive_runs[i]}",
            f"{err_naive}",
            f"{wa_runs[i]}",
            cache_mark,
        )
    console.print(t)

    naive_total_eps = 5 * eps
    wa_total_eps = eps  # only the first call paid
    naive_distinct = len(set(naive_runs))
    wa_distinct = len(set(wa_runs))

    msg = (
        f"[bold]Naive DP[/bold]\n"
        f"  - distinct answers across 5 runs: [red]{naive_distinct}[/red] (fresh noise each call)\n"
        f"  - total privacy budget used:       [red]{naive_total_eps:.2f}[/red]\n\n"
        f"[bold]Workload-Aware (Ours)[/bold]\n"
        f"  - distinct answers across 5 runs: [bold green]{wa_distinct}[/bold green] "
        f"(noise added once, then served from cache)\n"
        f"  - total privacy budget used:       [bold green]{wa_total_eps:.2f}[/bold green] "
        f"({(1 - wa_total_eps/naive_total_eps)*100:.0f}% savings)\n\n"
        f"[bold yellow]>>> The Naive answers vary because Laplace noise is genuinely "
        f"randomized.[/bold yellow]\n"
        f"[bold yellow]>>> The Workload-Aware answers are identical because, by the "
        f"post-processing property of DP, returning a cached noisy result is FREE.[/bold yellow]"
    )
    console.print(Panel(msg, border_style="magenta",
                        title="Why this matters"))
    pause(auto)


def scenario_semantic(config, auto: bool):
    """Show semantic matching (Tree Kernel + AST Embedding) extending the cache."""
    title("SCENARIO: Semantic Matching catches what syntactic hashing misses")
    narration(
        "L1 cache (syntactic): matches identical queries.\n"
        "L2 cache (semantic):  Tree Kernel + AST Embedding catch structurally\n"
        "                     similar queries with different literal values.\n\n"
        "We run 5 parametric queries -- same template, different WHERE literals.\n"
        "Workload-Aware (L1): only the FIRST repeat hits; different params miss.\n"
        "Semantic-Aware (L1+L2): structurally similar queries hit L2 at eps=0.\n\n"
        "TRADEOFF: L2 returns the cached answer to the NEAREST query, which is\n"
        "approximate. We show both budget savings AND answer fidelity below."
    )
    pause(auto)

    queries = [
        "SELECT COUNT(*) FROM adult WHERE age > 30",
        "SELECT COUNT(*) FROM adult WHERE age > 40",
        "SELECT COUNT(*) FROM adult WHERE age > 50",
        "SELECT COUNT(*) FROM adult WHERE age > 60",
        "SELECT COUNT(*) FROM adult WHERE age > 70",
    ]

    # Truth
    mw_exact = DPMiddleware(config, mode=ExecutionMode.EXACT)
    truths = [mw_exact.execute(q).rows[0][0] for q in queries]

    # Workload-aware (L1 only)
    mw_wa = DPMiddleware(config, mode=ExecutionMode.WORKLOAD_DP)
    wa_results = [mw_wa.execute(q, epsilon=1.0) for q in queries]

    # Semantic (L1 + L2)
    mw_sem = DPMiddleware(config, mode=ExecutionMode.SEMANTIC_DP)
    sem_results = [mw_sem.execute(q, epsilon=1.0) for q in queries]

    from rich.table import Table
    t = Table(title="Parametric workload: same template, different age thresholds",
              box=box.DOUBLE_EDGE)
    t.add_column("Query", style="cyan", width=18)
    t.add_column("True", justify="right", style="green")
    t.add_column("WA (L1)", justify="right", style="yellow")
    t.add_column("WA cache?", justify="center")
    t.add_column("Sem (L1+L2)", justify="right", style="bold magenta")
    t.add_column("Sem hit?", justify="center")

    for i, q in enumerate(queries):
        short = q.split("WHERE")[1].strip()
        wa_hit = "[green]L1[/green]" if wa_results[i].cache_hit else "[dim]-[/dim]"
        sem_hit = "[bold green]L1/L2[/bold green]" if sem_results[i].cache_hit else "[dim]-[/dim]"
        t.add_row(
            short,
            f"{truths[i]}",
            f"{wa_results[i].rows[0][0]}",
            wa_hit,
            f"{sem_results[i].rows[0][0]}",
            sem_hit,
        )
    console.print(t)

    wa_sum = mw_wa.budget_summary()
    sem_sum = mw_sem.budget_summary()

    # Answer error analysis
    wa_errs = [abs(int(wa_results[i].rows[0][0]) - truths[i]) for i in range(len(queries))]
    sem_errs = [abs(int(sem_results[i].rows[0][0]) - truths[i]) for i in range(len(queries))]

    summary = Table(box=box.ROUNDED)
    summary.add_column("Metric", style="bold")
    summary.add_column("Workload-Aware (L1)", style="yellow", justify="right")
    summary.add_column("Semantic (L1+L2)", style="bold magenta", justify="right")
    summary.add_row("Cache hits / queries",
                    f"{wa_sum['cache_hits']} / {wa_sum['total_queries']}",
                    f"{sem_sum['cache_hits']} / {sem_sum['total_queries']}"
                    f" (exact={sem_sum['exact_hits']}, sem={sem_sum['semantic_hits']})")
    summary.add_row("Epsilon consumed",
                    f"{wa_sum['consumed_epsilon']:.2f}",
                    f"{sem_sum['consumed_epsilon']:.2f}")
    summary.add_row("Mean abs answer error",
                    f"{sum(wa_errs)/len(wa_errs):.0f}",
                    f"{sum(sem_errs)/len(sem_errs):.0f}")
    console.print(summary)

    msg = (
        f"[bold]Budget savings:[/bold] "
        f"L1 used [yellow]{wa_sum['consumed_epsilon']:.0f}[/yellow] eps, "
        f"L1+L2 used [bold magenta]{sem_sum['consumed_epsilon']:.0f}[/bold magenta] eps "
        f"([bold green]{(1 - sem_sum['consumed_epsilon']/max(wa_sum['consumed_epsilon'], 0.001))*100:.0f}% "
        f"additional savings via semantic match[/bold green]).\n\n"
        f"[bold]Answer fidelity tradeoff:[/bold] L2 returns the cached answer to\n"
        f"the nearest cached query. For repetitive parametric workloads this is\n"
        f"often acceptable (dashboards, drill-downs). For exact accuracy, L1 only."
    )
    console.print(Panel(msg, border_style="magenta",
                        title="Semantic matching: when budget matters more than per-query exactness"))
    pause(auto)


def scenario_4(config, auto: bool):
    title("SCENARIO 4: Workload-level budget savings on TPC-H (SF=1)")
    narration(
        "We run a 20-query repetitive workload on TPC-H lineitem (6M rows).\n"
        "Both modes start with the same total budget (eps = 10).\n"
        "Naive consumes 1.0 per query and exhausts after 10 queries.\n"
        "Workload-aware caches and answers ALL 20 queries within budget."
    )
    pause(auto)

    sql = "SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'"
    n_queries = 20

    # Naive
    mw_naive = DPMiddleware(config, mode=ExecutionMode.NAIVE_DP)
    naive_results = []
    for i in range(n_queries):
        r = mw_naive.execute(sql, epsilon=1.0)
        naive_results.append(r)
    naive_summary = mw_naive.budget_summary()

    # Workload-aware
    mw_wa = DPMiddleware(config, mode=ExecutionMode.WORKLOAD_DP)
    wa_results = []
    for i in range(n_queries):
        r = mw_wa.execute(sql, epsilon=1.0)
        wa_results.append(r)
    wa_summary = mw_wa.budget_summary()

    naive_answered = sum(1 for r in naive_results if r.error is None)
    wa_answered = sum(1 for r in wa_results if r.error is None)
    wa_cache_hits = sum(1 for r in wa_results if r.cache_hit)

    from rich.table import Table
    t = Table(title=f"20 Repeated COUNT Queries (Budget: {config.privacy.total_epsilon} eps total)",
              box=box.DOUBLE_EDGE)
    t.add_column("Metric", style="bold cyan")
    t.add_column("PINQ / Naive DP", style="red", justify="right")
    t.add_column("Workload-Aware (Ours)", style="bold green", justify="right")

    t.add_row("Total queries answered",
              f"{naive_answered} / {n_queries}",
              f"{wa_answered} / {n_queries}")
    t.add_row("Cache hits", "0", f"{wa_cache_hits}")
    t.add_row("Total eps consumed",
              f"{naive_summary['consumed_epsilon']:.2f}",
              f"{wa_summary['consumed_epsilon']:.2f}")
    t.add_row("Budget remaining",
              f"{naive_summary['remaining_epsilon']:.2f}",
              f"{wa_summary['remaining_epsilon']:.2f}")
    t.add_row("Queries denied",
              f"{n_queries - naive_answered}",
              f"{n_queries - wa_answered}")

    console.print(t)

    savings_pct = (1 - wa_summary['consumed_epsilon'] /
                   max(naive_summary['consumed_epsilon'], 0.001)) * 100
    console.print(Panel(
        f"[bold green]>>> Workload-Aware answers {wa_answered - naive_answered}x more queries "
        f"with {savings_pct:.0f}% less budget. <<<[/bold green]",
        border_style="green"))
    pause(auto)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true",
                        help="Run all scenarios without pauses")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--scenario", type=int, default=None,
                        help="Run only one scenario (1-4)")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    title("DP-SQL Middleware: Project Demo")
    console.print(
        "[bold cyan]A Differentially Private SQL Middleware\n"
        "for Repeated Aggregate Queries[/bold cyan]\n\n"
        "Author: Refik Can Oztas | CMP653 Project | Hacettepe University\n"
        f"Backend: DuckDB | Data: TPC-H SF=1 + UCI Adult (48,842 rows)\n"
    )
    pause(args.auto)

    scenarios = {
        1: scenario_1,
        2: scenario_2,
        3: scenario_3,
        4: scenario_noise_visible,
        5: scenario_4,
        6: scenario_semantic,
    }
    if args.scenario:
        scenarios[args.scenario](config, args.auto)
    else:
        for s in [1, 2, 3, 4, 5, 6]:
            scenarios[s](config, args.auto)

    title("END OF DEMO")
    console.print(
        "[bold green]Key Takeaways:[/bold green]\n"
        " 1. Aggregate queries leak privacy via differencing/reconstruction attacks.\n"
        " 2. Differential privacy adds calibrated Laplace noise.\n"
        " 3. Naive composition wastes budget on repeated queries.\n"
        " 4. Workload-aware caching saves [bold]up to 95%[/bold] budget on repetitive workloads.\n"
        " 5. Caching is privacy-safe by the post-processing property of DP.\n"
    )


if __name__ == "__main__":
    main()
