"""
06_export_for_r.py
Export raw counts for R's statistical tests. For each topic: how many
questions had a signal, and how many didn't. Raw counts are needed because
R's chi-square test requires integer cell values, not pre-calculated rates.

Exports both overall contingency tables (used by existing R stats scripts)
and per-category contingency tables in long format (used by the new
03_stats_by_category.R script).
"""

import os
import duckdb

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH     = os.path.join(ROOT, "data", "db", "so.duckdb")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Same threshold as 05_aggregate.py, so the output covers the same topics.
MIN_QUESTIONS = 1000

def _category_names_from_db(con: duckdb.DuckDBPyConnection) -> list[str]:
    cols = con.execute("DESCRIBE question_miscomm").fetchdf()["column_name"].tolist()
    return [c[len("cat_"):] for c in cols if c.startswith("cat_")]

def export_contingency_tags(con: duckdb.DuckDBPyConnection):
    """Overall contingency table for Track A (tags)."""
    df = con.execute(f"""
        SELECT
            t.tag                                    AS topic,
            SUM(m.has_miscomm::INTEGER)              AS n_miscomm,
            SUM((NOT m.has_miscomm)::INTEGER)        AS n_no_miscomm,
            COUNT(*)                                 AS n_total
        FROM question_tags t
        JOIN question_miscomm m ON t.question_id = m.question_id
        GROUP BY t.tag
        HAVING COUNT(*) >= {MIN_QUESTIONS}
        ORDER BY n_total DESC
    """).fetchdf()

    out = os.path.join(RESULTS_DIR, "contingency_tags.csv")
    df.to_csv(out, index=False)
    print(f"contingency_tags.csv: {len(df)} rows → {out}")

def export_contingency_tags_by_category(con: duckdb.DuckDBPyConnection, cats: list[str]):
    """Per-category contingency table for Track A (tags), long format.

    Columns: topic, category, n_miscomm, n_no_miscomm, n_total
    One row per (tag, category) pair. n_total is constant across categories
    for a given tag (total questions with that tag).
    """
    import pandas as pd
    rows = []
    for cat in cats:
        df = con.execute(f"""
            SELECT
                t.tag                                    AS topic,
                SUM(m.cat_{cat}::INTEGER)                AS n_miscomm,
                SUM((NOT m.cat_{cat})::INTEGER)          AS n_no_miscomm,
                COUNT(*)                                 AS n_total
            FROM question_tags t
            JOIN question_miscomm m ON t.question_id = m.question_id
            GROUP BY t.tag
            HAVING COUNT(*) >= {MIN_QUESTIONS}
            ORDER BY n_total DESC
        """).fetchdf()
        df.insert(1, "category", cat)
        rows.append(df)

    combined = pd.concat(rows, ignore_index=True)
    out = os.path.join(RESULTS_DIR, "contingency_tags_by_category.csv")
    combined.to_csv(out, index=False)
    print(f"contingency_tags_by_category.csv: {len(combined)} rows → {out}")

if __name__ == "__main__":
    con  = duckdb.connect(DB_PATH)
    cats = _category_names_from_db(con)

    export_contingency_tags(con)
    export_contingency_tags_by_category(con, cats)

    con.close()
    print("Phase 7 complete.")
