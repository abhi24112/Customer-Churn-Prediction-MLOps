import os
import psycopg2
from psycopg2 import sql, extras


def _get_dsn():
    # psycopg2 accepts a libpq-style DSN, including a URL.
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url

    # Fallback (common in docker/CI)
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "postgres")
    dbname = os.getenv("PGDATABASE", "churn")

    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def get_connection():
    return psycopg2.connect(_get_dsn())

def ensure_schema(conn, schema: str) -> None:
    if not schema or schema == "public":
        return
    with conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    conn.commit()


def insert_rows(table_name: str, rows, schema: str = "public") -> int:

    if isinstance(rows, dict):
        rows = [rows]

    if not rows:
        raise ValueError("rows must not be empty")

    columns = list(rows[0].keys())
    values = [tuple(row.get(column) for column in columns) for row in rows]

    table_ref = sql.Identifier(schema, table_name)
    column_list = sql.SQL(", ").join(sql.Identifier(column) for column in columns)
    query = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(table_ref, column_list)

    with get_connection() as conn:
        with conn.cursor() as cur:
            extras.execute_values(cur, query.as_string(conn), values)
        conn.commit()

    return len(values)


def insert_row(table_name: str, row: dict, schema: str = "public") -> int:
    """Insert a single row into a table."""
    return insert_rows(table_name, row, schema=schema)
