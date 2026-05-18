"""Database backend abstraction. Supports PostgreSQL and DuckDB."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from dpdb.config import DBConfig


class Database(ABC):
    @abstractmethod
    def connect(self): ...
    @abstractmethod
    def close(self): ...
    @abstractmethod
    def execute(self, sql: str, params: tuple = ()) -> list[tuple]: ...
    @abstractmethod
    def execute_scalar(self, sql: str, params: tuple = ()) -> Any: ...
    @abstractmethod
    def execute_with_columns(self, sql: str, params: tuple = ()) -> tuple[list[str], list[tuple]]: ...

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()


class PostgresDatabase(Database):
    def __init__(self, config: DBConfig):
        self.config = config
        self._conn = None

    def connect(self):
        import psycopg2
        self._conn = psycopg2.connect(
            host=self.config.host, port=self.config.port,
            dbname=self.config.name, user=self.config.user,
            password=self.config.password,
        )
        self._conn.autocommit = True

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self.connect()
        return self._conn

    def execute(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description is None:
                return []
            return cur.fetchall()

    def execute_scalar(self, sql: str, params: tuple = ()) -> Any:
        rows = self.execute(sql, params)
        return rows[0][0] if rows and rows[0] else None

    def execute_with_columns(self, sql: str, params: tuple = ()) -> tuple[list[str], list[tuple]]:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description is None:
                return [], []
            columns = [desc[0] for desc in cur.description]
            return columns, cur.fetchall()


class DuckDBDatabase(Database):
    """DuckDB backend. Uses a local file; no server required."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None

    def connect(self):
        import duckdb
        self._conn = duckdb.connect(self.db_path, read_only=True)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            self.connect()
        return self._conn

    def execute(self, sql: str, params: tuple = ()) -> list[tuple]:
        cur = self.conn.execute(sql, params) if params else self.conn.execute(sql)
        try:
            return cur.fetchall()
        except Exception:
            return []

    def execute_scalar(self, sql: str, params: tuple = ()) -> Any:
        rows = self.execute(sql, params)
        return rows[0][0] if rows and rows[0] else None

    def execute_with_columns(self, sql: str, params: tuple = ()) -> tuple[list[str], list[tuple]]:
        cur = self.conn.execute(sql, params) if params else self.conn.execute(sql)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        try:
            rows = cur.fetchall()
        except Exception:
            rows = []
        return columns, rows


def create_database(config) -> Database:
    """Factory: create appropriate backend based on config."""
    backend = getattr(config, "backend", "duckdb")
    if backend == "postgres":
        return PostgresDatabase(config.db)
    elif backend == "duckdb":
        db_path = getattr(config, "duckdb_path", "data/dpdb.duckdb")
        # Resolve relative path from project root
        if not Path(db_path).is_absolute():
            db_path = str(Path(__file__).parent.parent.parent / db_path)
        return DuckDBDatabase(db_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")
