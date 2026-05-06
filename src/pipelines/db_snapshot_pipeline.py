"""Export a snapshot CSV from PostgreSQL for DVC.

Architecture:
  Postgres -> snapshot CSV -> DVC pipeline (preprocess/train)

This script overwrites a single file so your existing ingestion code
(`reading_files()` expects exactly one CSV/XLSX in the directory).

Env vars:
  DATABASE_URL (required unless PG* vars are set)
  DB_SCHEMA (default: public)
  RAW_TABLE (default: churn_raw)
  SNAPSHOT_CSV_PATH (default: src/data/raw_data/data.csv)

Notes:
  - DVC cannot automatically detect DB changes. Use `dvc repro -f db_snapshot`
    (Airflow DAG is updated to force this stage).
"""

import os
import pandas as pd
from backend.postgres import get_connection


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def main() -> None:
    schema = os.getenv("DB_SCHEMA", "public")
    table = os.getenv("RAW_TABLE", "churn_raw")
    out_path = os.getenv("SNAPSHOT_CSV_PATH", "src/data/raw_data/data.csv")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fq_table = f"{_quote_ident(schema)}.{_quote_ident(table)}" if schema else _quote_ident(table)

    with get_connection() as conn:
        df = pd.read_sql_query(f"SELECT * FROM {fq_table}", con=conn) # type: ignore

    if df.empty:
        raise ValueError(f"No rows found in {schema}.{table}")

    df.to_csv(out_path, index=False)
    print(f"Snapshot exported: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
