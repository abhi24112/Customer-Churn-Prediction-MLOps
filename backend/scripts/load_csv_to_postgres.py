"""Load a Kaggle CSV into PostgreSQL (simple bootstrap).

This creates/overwrites a raw table and loads the CSV into it.

Usage (PowerShell example):
  $env:DATABASE_URL='postgresql://postgres:postgres@localhost:5432/churn'
  $env:RAW_CSV_PATH='src/data/raw_data/data.csv'
  python -m backend.scripts.load_csv_to_postgres

Env vars:
  DATABASE_URL (required unless PG* vars are set)
  RAW_CSV_PATH (default: src/data/raw_data/data.csv)
"""

from __future__ import annotations

import os

import pandas as pd
from psycopg2 import sql

from backend.postgres import get_connection


def main() -> None:
    csv_path = os.getenv("RAW_CSV_PATH", "src/data/raw_data/data.csv")
    table = "churn_raw"
    schema = "public"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df_head = pd.read_csv(csv_path, nrows=0)
    cols = df_head.columns.tolist()
    if not cols:
        raise ValueError("CSV has no columns")

    # Keep it simple: store everything as TEXT.
    # Snapshot export will recreate the CSV; pandas will infer types again later.
    col_defs = sql.SQL(", ").join(
        sql.Composed([sql.Identifier(c), sql.SQL(" TEXT")]) for c in cols
    )

    fq_table = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {} ").format(fq_table))
            cur.execute(sql.SQL("CREATE TABLE {} ({})").format(fq_table, col_defs))
        conn.commit()

        # COPY is fast and simple.
        with conn.cursor() as cur, open(csv_path, "r", encoding="utf-8") as f:
            copy_sql = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)").format(
                fq_table,
                sql.SQL(", ").join(sql.Identifier(c) for c in cols),
            )
            cur.copy_expert(copy_sql.as_string(conn), f)
        conn.commit()

    print(f"Loaded CSV into {schema}.{table} ({len(cols)} columns)")


if __name__ == "__main__":
    main()
