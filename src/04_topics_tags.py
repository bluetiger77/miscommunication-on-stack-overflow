"""
04_topics_tags.py
Parse Stack Overflow tags, expand each question's tags into separate rows,
and save to DuckDB. Produces the question_tags table.
"""

import os
import re
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH     = os.path.join(ROOT, "data", "db", "so.duckdb")
PARQUET_DIR = os.path.join(ROOT, "data", "parquet")
BATCH_SIZE  = 200_000

# Tags with fewer questions than this are excluded — too few to draw reliable conclusions.
TOP_N_TAGS = 200

# Stack Overflow encodes a question's tags as a string of angle-bracket-delimited
# tokens, e.g. "<python><pandas><dataframe>". This pattern extracts the name
# from each token. It matches any sequence of non-> characters between < and >.
TAG_RE = re.compile(r"<([^>]+)>")

def parse_tags(raw: str) -> list[str]:
    # Return the list of tag strings extracted from the raw "<tag1><tag2>" format.
    # Returns an empty list if raw is None or empty, so callers don't need to guard.
    if not raw:
        return []
    return TAG_RE.findall(raw)

def explode_tags(con: duckdb.DuckDBPyConnection):
    """
    Reads questions in batches, parses each question's raw tag string into
    individual tags, and writes one (question_id, tag) row per each tag for the question to a
    Parquet file at PARQUET_DIR/question_tags.parquet.
    """
    out_path = os.path.join(PARQUET_DIR, "question_tags.parquet")

    # One row per (question_id, tag) pair — a question with 5 tags produces 5 rows.
    schema = pa.schema([
        ("question_id", pa.int64()),
        ("tag",         pa.utf8()),
    ])
    writer = pq.ParquetWriter(out_path, schema, compression="snappy")

    total = con.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    print(f"Exploding tags for {total:,} questions ...")

    offset = 0
    processed = 0
    while True:
        # Fetch a batch of questions. ORDER BY ensures consistent ordering across runs.
        rows = con.execute(f"""
            SELECT question_id, raw_tags
            FROM questions
            ORDER BY question_id
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """).fetchall()
        if not rows:
            break

        # For each question, emit one (question_id, tag) pair per tag.
        # Both lists stay in sync for column construction.
        qids, tnames = [], []
        for question_id, raw_tags in rows:
            for tag in parse_tags(raw_tags):
                qids.append(question_id)
                tnames.append(tag)

        # Skip empty batches — PyArrow warns if you write zero rows.
        if qids:
            batch = pa.record_batch(
                [pa.array(qids, pa.int64()),
                 pa.array(tnames, pa.utf8())],
                schema=schema,
            )
            writer.write_batch(batch)

        offset += BATCH_SIZE
        processed += len(rows)
        print(f"  {processed:,}/{total:,}", end="\r", flush=True)

    writer.close()
    print(f"\n  → {out_path}")
    return out_path

def load_to_db(con: duckdb.DuckDBPyConnection, parquet_path: str):
    # Load only the top TOP_N_TAGS tags by question count. The subquery ranks
    # all tags by frequency; the JOIN keeps only rows for those tags.
    con.execute(f"""
        CREATE OR REPLACE TABLE question_tags AS
        SELECT qt.question_id, qt.tag
        FROM read_parquet('{parquet_path}') qt
        -- Filter to top-N tags only
        JOIN (
            SELECT tag
            FROM read_parquet('{parquet_path}')
            GROUP BY tag
            ORDER BY COUNT(*) DESC
            LIMIT {TOP_N_TAGS}
        ) top_tags ON qt.tag = top_tags.tag
    """)

    n_rows = con.execute("SELECT COUNT(*) FROM question_tags").fetchone()[0]
    n_tags = con.execute("SELECT COUNT(DISTINCT tag) FROM question_tags").fetchone()[0]
    print(f"question_tags: {n_rows:,} (question, tag) pairs across {n_tags} tags")

    # Save tag frequencies to CSV for reference — which tags made the cut and
    # how many questions each has.
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    tag_counts = con.execute("""
        SELECT tag, COUNT(*) AS n_questions
        FROM question_tags
        GROUP BY tag
        ORDER BY n_questions DESC
    """).fetchdf()
    tag_counts.to_csv(os.path.join(results_dir, "tag_counts.csv"), index=False)
    print(f"  tag_counts.csv written ({len(tag_counts)} tags)")

if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    parquet_path = explode_tags(con)
    load_to_db(con, parquet_path)
    con.close()
    print("Phase 4 complete.")
