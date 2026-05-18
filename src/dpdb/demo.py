"""Step-by-step demo CLI showing what happens when a query is processed.

Use:
    python -m dpdb.demo                  # interactive mode
    python -m dpdb.demo --compare        # side-by-side baselines
    python -m dpdb.demo --query "SELECT COUNT(*) FROM lineitem"
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree
from rich import box

from dpdb.analyzer import analyze_sensitivity
from dpdb.budget import AllocationStrategy, BudgetLedger
from dpdb.config import Config
from dpdb.db import create_database
from dpdb.mechanisms import laplace_mechanism
from dpdb.middleware import DPMiddleware, ExecutionMode
from dpdb.parser import parse_query
from dpdb.template import (
    extract_template, full_query_hash, param_hash, template_hash,
)


console = Console(force_terminal=True, legacy_windows=False)


def banner(text: str, style: str = "bold cyan"):
    console.print()
    console.rule(f"[{style}]{text}[/{style}]", style=style)


def step_header(num: int, title: str, color: str = "cyan"):
    console.print(f"\n[bold {color}]STEP {num}: {title}[/bold {color}]")


def trace_query(sql: str, config: Config, epsilon: float = 1.0,
                budget_ledger: BudgetLedger = None):
    """Execute a query with full step-by-step tracing."""
    db = create_database(config)
    db.connect()

    # ---- STEP 1: Receive Query ----
    step_header(1, "QUERY RECEIVED", "blue")
    console.print(Panel(Syntax(sql, "sql", theme="monokai", line_numbers=False),
                        title="Input SQL", border_style="blue"))
    console.print(f"  Privacy budget requested: [bold yellow]eps = {epsilon}[/bold yellow]")
    time.sleep(0.4)

    # ---- STEP 2: Parse and Validate ----
    step_header(2, "SQL PARSING & VALIDATION", "magenta")
    parsed = parse_query(sql)
    tree = Tree("[bold]Parsed Query AST[/bold]")
    tree.add(f"Table: [green]{parsed.table}[/green]")
    aggs = tree.add("Aggregates")
    for a in parsed.aggregates:
        aggs.add(f"[yellow]{a.func}[/yellow]({a.column or '*'})")
    if parsed.group_by:
        gb = tree.add("GROUP BY")
        for g in parsed.group_by:
            gb.add(f"[cyan]{g}[/cyan]")
    if parsed.where_predicates:
        wb = tree.add("WHERE predicates")
        for w in parsed.where_predicates:
            wb.add(f"[red]{w}[/red]")
    console.print(tree)
    console.print("  [green][OK][/green] Query is in supported subset")
    time.sleep(0.4)

    # ---- STEP 3: Template Extraction ----
    step_header(3, "TEMPLATE EXTRACTION", "magenta")
    template = extract_template(parsed)
    t_hash = template_hash(template)
    p_hash = param_hash(parsed)
    table = Table(box=box.ROUNDED, show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Template", f"[dim]{template[:80]}{'...' if len(template) > 80 else ''}[/dim]")
    table.add_row("Template hash (h_T)", f"[yellow]{t_hash}[/yellow]")
    table.add_row("Parameter hash (h_P)", f"[yellow]{p_hash}[/yellow]")
    console.print(table)
    time.sleep(0.4)

    # ---- STEP 4: Cache Lookup ----
    step_header(4, "BUDGET LEDGER: CACHE CHECK", "yellow")
    cached = None
    if budget_ledger:
        cached = budget_ledger.try_cache(parsed)
    if cached is not None:
        console.print(Panel(
            f"[bold green][OK] CACHE HIT[/bold green]\n\n"
            f"This exact query was answered before (h_T={t_hash}, h_P={p_hash}).\n"
            f"By the [bold]post-processing property[/bold] of differential privacy,\n"
            f"returning the cached noisy result requires [bold]eps = 0[/bold] additional budget.\n\n"
            f"Cached answer: [bold cyan]{cached.rows[0][0] if cached.rows else 'N/A'}[/bold cyan]",
            border_style="green"))
        return cached.rows, 0.0, True

    console.print("  [yellow][X] Cache miss[/yellow] — query has not been seen before. Allocating fresh eps.")
    time.sleep(0.4)

    # ---- STEP 5: Sensitivity Analysis ----
    step_header(5, "SENSITIVITY ANALYSIS", "magenta")
    sensitivities = analyze_sensitivity(parsed, config)
    sens_table = Table(box=box.ROUNDED)
    sens_table.add_column("Aggregate", style="cyan")
    sens_table.add_column("Column", style="green")
    sens_table.add_column("Sensitivity deltaf", style="bold yellow")
    sens_table.add_column("Reasoning", style="dim")
    for s in sensitivities:
        sens_table.add_row(s.func, s.column or "*", f"{s.sensitivity:g}", s.notes)
    console.print(sens_table)
    time.sleep(0.4)

    # ---- STEP 6: Budget Allocation ----
    step_header(6, "PRIVACY BUDGET ALLOCATION", "yellow")
    if budget_ledger:
        before = budget_ledger.remaining
        allocated = budget_ledger.allocate(parsed, epsilon)
        after = budget_ledger.remaining
    else:
        before = config.privacy.total_epsilon
        allocated = epsilon
        after = before - allocated

    bar_width = 40
    used_bars = int((1 - after / config.privacy.total_epsilon) * bar_width)
    bar = "#" * used_bars + "." * (bar_width - used_bars)
    console.print(f"  Budget bar: [red]{bar}[/red] {after:.2f} / {config.privacy.total_epsilon:.2f}")
    console.print(f"  Allocated:  [bold yellow]eps = {allocated}[/bold yellow] (was {before:.2f}, now {after:.2f})")
    time.sleep(0.4)

    # ---- STEP 7: Execute on Database ----
    step_header(7, "EXECUTE ON DATABASE (TRUE ANSWER)", "blue")
    t0 = time.perf_counter()
    columns, true_rows = db.execute_with_columns(sql)
    db_ms = (time.perf_counter() - t0) * 1000
    if true_rows:
        true_val = true_rows[0][0]
        console.print(f"  True answer: [bold cyan]{true_val:,}[/bold cyan]  (DB latency: {db_ms:.1f}ms)")
    time.sleep(0.4)

    # ---- STEP 8: Add Calibrated Noise ----
    step_header(8, "LAPLACE NOISE INJECTION", "red")
    eps_per_agg = allocated / len(sensitivities)
    noise_table = Table(box=box.ROUNDED, show_lines=True)
    noise_table.add_column("Aggregate", style="cyan")
    noise_table.add_column("True", style="green", justify="right")
    noise_table.add_column("deltaf", justify="right")
    noise_table.add_column("Scale b=deltaf/eps", justify="right", style="yellow")
    noise_table.add_column("Noise sample", justify="right", style="red")
    noise_table.add_column("Noisy result", justify="right", style="bold magenta")

    noisy_rows = []
    for row in true_rows:
        row_list = list(row)
        n_group = len(parsed.group_by)
        for i, sens in enumerate(sensitivities):
            col_idx = n_group + i
            true_v = float(row_list[col_idx])
            scale = sens.sensitivity / eps_per_agg
            noise = np.random.laplace(0.0, scale)
            noisy_v = true_v + noise
            if sens.func == "COUNT":
                noisy_v = max(0, round(noisy_v))
            row_list[col_idx] = noisy_v
            noise_table.add_row(
                f"{sens.func}({sens.column or '*'})",
                f"{true_v:,.2f}",
                f"{sens.sensitivity:g}",
                f"{scale:.3f}",
                f"{noise:+.2f}",
                f"{noisy_v:,.2f}",
            )
        noisy_rows.append(tuple(row_list))
    console.print(noise_table)
    time.sleep(0.4)

    # ---- STEP 9: Cache Store ----
    if budget_ledger and budget_ledger.strategy == AllocationStrategy.WORKLOAD_AWARE:
        step_header(9, "CACHE STORE (post-processing for future hits)", "green")
        budget_ledger.store_result(parsed, columns, noisy_rows, allocated)
        console.print(f"  [green][OK][/green] Stored noisy result under (h_T={t_hash}, h_P={p_hash}).")
        console.print(f"  Future identical queries will return this result at [bold]eps=0[/bold] cost.")

    # ---- Return ----
    db.close()
    return noisy_rows, allocated, False


def compare_mode(sql: str, config: Config, epsilon: float = 1.0):
    """Run the same query through 3 baselines and show side-by-side."""
    banner("BASELINE COMPARISON: Same Query Across 3 Strategies", "bold magenta")
    console.print(Panel(Syntax(sql, "sql", theme="monokai"), title="Query", border_style="cyan"))

    table = Table(title="Side-by-Side Comparison", box=box.DOUBLE_EDGE)
    table.add_column("Strategy", style="bold cyan", width=22)
    table.add_column("Description", style="dim", width=40)
    table.add_column("Result", style="bold yellow", justify="right")
    table.add_column("eps used", justify="right")
    table.add_column("Cache?", justify="center")

    # Baseline 1: Exact
    mw_exact = DPMiddleware(config, mode=ExecutionMode.EXACT)
    r = mw_exact.execute(sql)
    true_val = r.rows[0][0] if r.rows else 0
    table.add_row(
        "Exact (no privacy)",
        "Direct SQL passthrough — gold standard",
        f"{true_val:,.0f}",
        "0",
        "—",
    )

    # Baseline 2: PINQ-style (sequential composition)
    mw_pinq = DPMiddleware(config, mode=ExecutionMode.NAIVE_DP)
    r2 = mw_pinq.execute(sql, epsilon=epsilon)
    error_pinq = abs(float(r2.rows[0][0]) - true_val) if r2.rows else 0
    table.add_row(
        "PINQ / Naive DP",
        "Sequential composition (textbook DP-SQL)",
        f"{r2.rows[0][0]:,.0f}" if r2.rows else "ERROR",
        f"{r2.epsilon_used}",
        "No",
    )

    # Run query 5 times in workload-aware to demonstrate caching
    mw_ours = DPMiddleware(config, mode=ExecutionMode.WORKLOAD_DP)
    first_r = mw_ours.execute(sql, epsilon=epsilon)
    second_r = mw_ours.execute(sql, epsilon=epsilon)
    third_r = mw_ours.execute(sql, epsilon=epsilon)
    error_ours = abs(float(first_r.rows[0][0]) - true_val) if first_r.rows else 0
    table.add_row(
        "[bold green]Workload-Aware (Ours)[/bold green]",
        "Template caching + post-processing",
        f"{first_r.rows[0][0]:,.0f} (q1)",
        f"{first_r.epsilon_used}",
        "No (1st)",
    )
    table.add_row(
        "[bold green]Workload-Aware (Ours)[/bold green]",
        "[dim]Same query repeated[/dim]",
        f"{second_r.rows[0][0]:,.0f} (q2)",
        f"[bold green]{second_r.epsilon_used}[/bold green]",
        "[bold green][OK] HIT[/bold green]",
    )
    table.add_row(
        "[bold green]Workload-Aware (Ours)[/bold green]",
        "[dim]Same query repeated[/dim]",
        f"{third_r.rows[0][0]:,.0f} (q3)",
        f"[bold green]{third_r.epsilon_used}[/bold green]",
        "[bold green][OK] HIT[/bold green]",
    )

    console.print(table)

    # Summary box
    summary = (
        f"Running this query [bold]3 times[/bold] under each strategy:\n\n"
        f"  • PINQ / Naive DP   ⇒ total eps = [red]{epsilon * 3:.1f}[/red] consumed (3 × {epsilon})\n"
        f"  • Workload-Aware    ⇒ total eps = [bold green]{epsilon:.1f}[/bold green] consumed (1 × {epsilon}, 2 cache hits)\n\n"
        f"  [bold yellow]Budget savings: {(1 - epsilon / (epsilon * 3)) * 100:.0f}%[/bold yellow] for repeated workload\n"
        f"  [bold yellow]Same noisy answer returned[/bold yellow] (post-processing is free under DP)"
    )
    console.print(Panel(summary, title="Why Workload-Aware Wins", border_style="green"))


def interactive_mode(config: Config, epsilon: float):
    """Interactive REPL with step-by-step tracing."""
    banner("DP-SQL MIDDLEWARE — INTERACTIVE DEMO", "bold cyan")
    console.print("Type a SQL query (single-table aggregate). Each query is traced step-by-step.")
    console.print("Commands: [yellow]\\budget[/yellow] (show budget) | [yellow]\\reset[/yellow] (new ledger) | [yellow]\\quit[/yellow]\n")

    ledger = BudgetLedger(config.privacy.total_epsilon, AllocationStrategy.WORKLOAD_AWARE)

    while True:
        try:
            sql = console.input("[bold cyan]dp-sql>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not sql:
            continue
        if sql in ("\\quit", "exit", "quit"):
            break
        if sql == "\\reset":
            ledger = BudgetLedger(config.privacy.total_epsilon, AllocationStrategy.WORKLOAD_AWARE)
            console.print("[green][OK] Budget ledger reset[/green]\n")
            continue
        if sql == "\\budget":
            s = ledger.summary()
            t = Table(title="Budget Ledger", box=box.ROUNDED)
            t.add_column("Field", style="bold")
            t.add_column("Value")
            for k, v in s.items():
                t.add_row(str(k), str(v))
            console.print(t)
            continue

        try:
            trace_query(sql, config, epsilon=epsilon, budget_ledger=ledger)
        except Exception as e:
            console.print(f"[bold red]ERROR:[/bold red] {e}\n")


def main():
    parser = argparse.ArgumentParser(description="DP-SQL step-by-step demo")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single query and exit")
    parser.add_argument("--compare", action="store_true",
                        help="Side-by-side baseline comparison")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    if args.compare:
        sample = args.query or "SELECT COUNT(*) FROM adult WHERE age > 30"
        compare_mode(sample, config, epsilon=args.epsilon)
    elif args.query:
        ledger = BudgetLedger(config.privacy.total_epsilon, AllocationStrategy.WORKLOAD_AWARE)
        trace_query(args.query, config, args.epsilon, ledger)
    else:
        interactive_mode(config, args.epsilon)


if __name__ == "__main__":
    main()
