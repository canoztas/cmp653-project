"""Load TPC-H SF=10 into a separate DuckDB file for scale-factor validation."""
import duckdb
import os
import time

t0 = time.time()
path = "data/dpdb_sf10.duckdb"
if os.path.exists(path):
    os.remove(path)
    print(f"Removed existing {path}")

con = duckdb.connect(path)
print("Installing tpch extension...")
con.execute("INSTALL tpch")
con.execute("LOAD tpch")
print("Generating SF=10 data (this takes a few minutes)...")
con.execute("CALL dbgen(sf=10)")

for t in ["region", "nation", "supplier", "customer", "part",
          "partsupp", "orders", "lineitem"]:
    n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t}: {n:,}")

con.close()
print(f"Total time: {time.time() - t0:.1f}s")
