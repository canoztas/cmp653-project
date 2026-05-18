"""Generate TPC-H data via DuckDB and load Adult dataset.

This is the unified data loader. It produces a single DuckDB database file
containing both TPC-H (all 8 tables) at a configurable scale factor, and
the UCI Adult dataset loaded as a single table.

Output: data/dpdb.duckdb
"""

import argparse
import sys
from pathlib import Path

import duckdb


ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

ADULT_SCHEMA = """
CREATE TABLE adult (
    age INTEGER,
    workclass VARCHAR,
    fnlwgt INTEGER,
    education VARCHAR,
    education_num INTEGER,
    marital_status VARCHAR,
    occupation VARCHAR,
    relationship VARCHAR,
    race VARCHAR,
    sex VARCHAR,
    capital_gain INTEGER,
    capital_loss INTEGER,
    hours_per_week INTEGER,
    native_country VARCHAR,
    income VARCHAR
)
"""


def generate_tpch(con: duckdb.DuckDBPyConnection, scale_factor: float):
    """Generate TPC-H data using DuckDB's tpch extension."""
    print(f"[TPC-H] Installing and loading extension...")
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    print(f"[TPC-H] Generating data at SF={scale_factor}...")
    con.execute(f"CALL dbgen(sf={scale_factor})")

    tables = ["region", "nation", "supplier", "customer", "part",
              "partsupp", "orders", "lineitem"]
    print(f"[TPC-H] Tables generated:")
    for t in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {count:,} rows")


def load_adult(con: duckdb.DuckDBPyConnection, data_dir: Path):
    """Load the UCI Adult dataset from downloaded files."""
    adult_data = data_dir / "adult" / "adult.data"
    adult_test = data_dir / "adult" / "adult.test"

    if not adult_data.exists():
        print(f"[Adult] ERROR: {adult_data} not found. Download first.")
        return

    print(f"[Adult] Creating schema...")
    con.execute("DROP TABLE IF EXISTS adult")
    con.execute(ADULT_SCHEMA)

    # Load training data (adult.data)
    # Format: CSV, comma-separated, no header, ?=null
    print(f"[Adult] Loading {adult_data}...")
    con.execute(f"""
        INSERT INTO adult
        SELECT * FROM read_csv('{adult_data.as_posix()}',
            header=false,
            columns={{
                'age': 'INTEGER',
                'workclass': 'VARCHAR',
                'fnlwgt': 'INTEGER',
                'education': 'VARCHAR',
                'education_num': 'INTEGER',
                'marital_status': 'VARCHAR',
                'occupation': 'VARCHAR',
                'relationship': 'VARCHAR',
                'race': 'VARCHAR',
                'sex': 'VARCHAR',
                'capital_gain': 'INTEGER',
                'capital_loss': 'INTEGER',
                'hours_per_week': 'INTEGER',
                'native_country': 'VARCHAR',
                'income': 'VARCHAR'
            }},
            nullstr='?',
            skip=0
        )
    """)

    # Load test data (skip the first line which is a comment)
    if adult_test.exists():
        print(f"[Adult] Loading {adult_test}...")
        con.execute(f"""
            INSERT INTO adult
            SELECT * FROM read_csv('{adult_test.as_posix()}',
                header=false,
                columns={{
                    'age': 'INTEGER',
                    'workclass': 'VARCHAR',
                    'fnlwgt': 'INTEGER',
                    'education': 'VARCHAR',
                    'education_num': 'INTEGER',
                    'marital_status': 'VARCHAR',
                    'occupation': 'VARCHAR',
                    'relationship': 'VARCHAR',
                    'race': 'VARCHAR',
                    'sex': 'VARCHAR',
                    'capital_gain': 'INTEGER',
                    'capital_loss': 'INTEGER',
                    'hours_per_week': 'INTEGER',
                    'native_country': 'VARCHAR',
                    'income': 'VARCHAR'
                }},
                nullstr='?',
                skip=1
            )
        """)

    count = con.execute("SELECT COUNT(*) FROM adult").fetchone()[0]
    print(f"[Adult] Loaded {count:,} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sf", type=float, default=1.0,
                        help="TPC-H scale factor (default 1.0)")
    parser.add_argument("--db", default="data/dpdb.duckdb",
                        help="Output DuckDB file path")
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing adult/ subdirectory")
    parser.add_argument("--skip-tpch", action="store_true")
    parser.add_argument("--skip-adult", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    db_path = root / args.db
    data_dir = root / args.data_dir

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        print(f"Removing existing {db_path}")
        db_path.unlink()

    con = duckdb.connect(str(db_path))

    if not args.skip_tpch:
        generate_tpch(con, args.sf)

    if not args.skip_adult:
        load_adult(con, data_dir)

    # Summary
    print("\n=== Database Summary ===")
    tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name").fetchall()
    for (t,) in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {count:,} rows")

    con.close()
    print(f"\nDatabase saved to: {db_path}")


if __name__ == "__main__":
    main()
