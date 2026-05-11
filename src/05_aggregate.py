"""
Join miscomm flags with topic assignments to produce
per-topic miscommunication rate tables. Exports CSVs for R.

Overall rates (has_miscomm) and per-category rates are both exported.
"""

import os
import csv
import duckdb

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH     = os.path.join(ROOT, "data", "db", "so.duckdb")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Topics with fewer questions than this are excluded — too few to give reliable rates.
MIN_QUESTIONS = 1000

def _category_names_from_db(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Derive category names from the cat_* columns present in question_miscomm."""
    cols = con.execute("DESCRIBE question_miscomm").fetchdf()["column_name"].tolist()
    return [c[len("cat_"):] for c in cols if c.startswith("cat_")]

def aggregate_tags(con: duckdb.DuckDBPyConnection):
    """Community tags (multi-label). Overall miscommunication rate."""
    df = con.execute(f"""
        SELECT
            t.tag                        AS topic,
            COUNT(*)                     AS n_questions,
            SUM(m.has_miscomm::INTEGER)  AS n_miscomm,
            AVG(m.has_miscomm::INTEGER)  AS miscomm_rate
        FROM question_tags t
        JOIN question_miscomm m ON t.question_id = m.question_id
        GROUP BY t.tag
        HAVING COUNT(*) >= {MIN_QUESTIONS}
        ORDER BY miscomm_rate DESC
    """).fetchdf()

    out = os.path.join(RESULTS_DIR, "track_a_tags.csv")
    df.to_csv(out, index=False)
    print(f"Tags: {len(df)} tags → {out}")
    print(df.head(10).to_string(index=False))
    return df

def aggregate_tags_by_category(con: duckdb.DuckDBPyConnection, cats: list[str]):
    """Per-category miscommunication rates, long format.

    Columns: topic, category, n_questions, n_miscomm, miscomm_rate
    n_questions is the same for every category within a tag (total questions
    carrying that tag), so each (tag, category) pair gives an independent rate.
    """
    rows = []
    for cat in cats:
        df = con.execute(f"""
            SELECT
                t.tag                              AS topic,
                COUNT(*)                           AS n_questions,
                SUM(m.cat_{cat}::INTEGER)          AS n_miscomm,
                AVG(m.cat_{cat}::INTEGER)          AS miscomm_rate
            FROM question_tags t
            JOIN question_miscomm m ON t.question_id = m.question_id
            GROUP BY t.tag
            HAVING COUNT(*) >= {MIN_QUESTIONS}
        """).fetchdf()
        df.insert(1, "category", cat)
        rows.append(df)

    import pandas as pd
    combined = pd.concat(rows, ignore_index=True)
    out = os.path.join(RESULTS_DIR, "track_a_tags_by_category.csv")
    combined.to_csv(out, index=False)
    print(f"Tags by category: {len(combined)} rows → {out}")
    return combined

def global_summary(con: duckdb.DuckDBPyConnection, cats: list[str]):
    """Overall miscommunication rate across all questions (overall + per category)."""
    row = con.execute("""
        SELECT
            COUNT(*)                     AS total_questions,
            SUM(has_miscomm::INTEGER)    AS total_miscomm,
            AVG(has_miscomm::INTEGER)    AS global_miscomm_rate
        FROM question_miscomm
    """).fetchone()
    total, flagged, rate = row

    total_answers  = con.execute("SELECT COUNT(*) FROM posts WHERE PostTypeId = 2").fetchone()[0]
    total_comments = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    total_tags     = con.execute("SELECT COUNT(*) FROM tags").fetchone()[0]

    print(f"\nGlobal: {total:,} questions, {flagged:,} flagged ({rate*100:.3f}%)")
    print(f"        {total_answers:,} answers, {total_comments:,} comments, {total_tags:,} tags")

    with open(os.path.join(RESULTS_DIR, "global_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["total_questions", "total_answers", "total_comments", "total_tags", "total_miscomm", "global_miscomm_rate"])
        w.writerow([total, total_answers, total_comments, total_tags, flagged, rate])

    # Per-category global baselines — used by R for per-category proportion tests.
    with open(os.path.join(RESULTS_DIR, "global_summary_by_category.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "total_questions", "total_miscomm", "global_miscomm_rate"])
        for cat in cats:
            cat_row = con.execute(f"""
                SELECT
                    COUNT(*)                    AS total_questions,
                    SUM(cat_{cat}::INTEGER)     AS total_miscomm,
                    AVG(cat_{cat}::INTEGER)     AS global_miscomm_rate
                FROM question_miscomm
            """).fetchone()
            w.writerow([cat, cat_row[0], cat_row[1], cat_row[2]])
            print(f"  {cat}: {cat_row[1]:,}/{cat_row[0]:,} ({cat_row[2]*100:.3f}%)")

if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    cats = _category_names_from_db(con)

    global_summary(con, cats)
    aggregate_tags(con)
    aggregate_tags_by_category(con, cats)

    con.close()
    print("\nPhase 6 complete.")
