"""Interactive CLI for the DP middleware."""

import argparse
import sys
from pathlib import Path

from dpdb.config import Config
from dpdb.middleware import DPMiddleware, ExecutionMode


def main():
    parser = argparse.ArgumentParser(description="DP-SQL Middleware CLI")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--mode",
        choices=["exact", "naive_dp", "workload_dp"],
        default="workload_dp",
    )
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Per-query epsilon (overrides config)")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    mode = ExecutionMode(args.mode)
    mw = DPMiddleware(config, mode=mode)

    print(f"DP-SQL Middleware | Mode: {mode.value} | Total budget: {config.privacy.total_epsilon}")
    print("Type SQL queries (Ctrl+D to quit). Supported: COUNT, SUM, AVG with GROUP BY/WHERE.\n")

    try:
        while True:
            remaining = mw.remaining_budget()
            prompt = f"[eps={remaining:.2f}] dp-sql> " if remaining != float("inf") else "dp-sql> "
            try:
                sql = input(prompt).strip()
            except EOFError:
                break

            if not sql:
                continue
            if sql.lower() in ("quit", "exit", "\\q"):
                break
            if sql.lower() == "\\budget":
                summary = mw.budget_summary()
                if summary:
                    for k, v in summary.items():
                        print(f"  {k}: {v}")
                else:
                    print("  No budget tracking (exact mode)")
                continue

            result = mw.execute(sql, epsilon=args.epsilon)

            if result.error:
                print(f"ERROR: {result.error}")
                continue

            # Print results in tabular format
            if result.columns:
                header = " | ".join(f"{c:>15}" for c in result.columns)
                print(header)
                print("-" * len(header))
                for row in result.rows:
                    vals = []
                    for v in row:
                        if isinstance(v, float):
                            vals.append(f"{v:>15.2f}")
                        else:
                            vals.append(f"{str(v):>15}")
                    print(" | ".join(vals))

            meta = []
            if result.epsilon_used > 0:
                meta.append(f"eps={result.epsilon_used:.4f}")
            if result.cache_hit:
                meta.append("CACHE HIT")
            meta.append(f"{result.latency_ms:.1f}ms")
            print(f"  ({', '.join(meta)})\n")

    except KeyboardInterrupt:
        pass

    print("\nFinal budget summary:")
    summary = mw.budget_summary()
    if summary:
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
