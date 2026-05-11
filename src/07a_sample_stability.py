"""
Analysis for when the concreteness scores stabilize for various SAMPLE_SIZE sizes in 07_abstractness.py.

For each of the top N_TAGS tags, repeatedly samples question titles at
increasing sample sizes and records the mean concreteness score. The goal
is to find the smallest sample size at which the mean stabilizes, justifying
the SAMPLE_SIZE constant chosen in 07_abstractness.py.

Outputs:
  results/sample_stability.csv  (tag, sample_size, run, n_words_matched, concreteness_score)
  Printed summary: mean ± std per (tag, sample_size)
"""

import importlib.util
import os
import duckdb
import pandas as pd

# --- Reuse functions from 07_abstractness.py ---
_spec = importlib.util.spec_from_file_location(
    "abstractness",
    os.path.join(os.path.dirname(__file__), "07_abstractness.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_norms   = _mod.load_norms
score_titles = _mod.score_titles

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH     = os.path.join(ROOT, "data", "db", "so.duckdb")
NORMS_PATH  = os.path.join(ROOT, "data", "concreteness_ratings.xlsx")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLE_SIZES  = [50, 100, 200, 300, 400, 500, 750, 1000]
N_RUNS        = 10   # repeated draws per (tag, sample_size) to measure variance
N_TAGS        = 5    # number of top tags to test
MIN_QUESTIONS = 1000


def run_stability(con: duckdb.DuckDBPyConnection, norms: dict) -> pd.DataFrame:
    # Top N_TAGS tags by question count.
    tags = con.execute(f"""
        SELECT tag
        FROM question_tags
        GROUP BY tag
        HAVING COUNT(*) >= {MIN_QUESTIONS}
        ORDER BY COUNT(*) DESC
        LIMIT {N_TAGS}
    """).fetchdf()["tag"].tolist()

    print(f"Testing stability for {N_TAGS} tags × {len(SAMPLE_SIZES)} sample sizes × {N_RUNS} runs ...")

    rows = []
    total = N_TAGS * len(SAMPLE_SIZES) * N_RUNS
    done = 0

    for tag in tags:
        for sample_size in SAMPLE_SIZES:
            for run in range(1, N_RUNS + 1):
                titles = con.execute("""
                    SELECT q.title
                    FROM question_tags t
                    JOIN questions q ON t.question_id = q.question_id
                    WHERE t.tag = ?
                      AND q.title IS NOT NULL
                    ORDER BY random()
                    LIMIT ?
                """, [tag, sample_size]).fetchdf()["title"].tolist()

                n_matched, score = score_titles(titles, norms)
                rows.append({
                    "tag":               tag,
                    "sample_size":       sample_size,
                    "run":               run,
                    "n_words_matched":   n_matched,
                    "concreteness_score": score,
                })
                done += 1
                print(f"  {done}/{total}  {tag}  n={sample_size}  run={run}  score={f'{score:.3f}' if score else 'N/A'}",
                      end="\r", flush=True)

    print()
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    summary = (
        df.dropna(subset=["concreteness_score"])
          .groupby(["tag", "sample_size"])["concreteness_score"]
          .agg(mean="mean", std="std")
          .reset_index()
    )
    print(f"\n{'tag':<20} {'n':>6}  {'mean':>7}  {'std':>7}")
    print("-" * 46)
    current_tag = None
    for _, row in summary.iterrows():
        if row["tag"] != current_tag:
            if current_tag is not None:
                print()
            current_tag = row["tag"]
        print(f"{row['tag']:<20} {row['sample_size']:>6}  {row['mean']:>7.4f}  {row['std']:>7.4f}")


if __name__ == "__main__":
    if not os.path.exists(NORMS_PATH):
        raise FileNotFoundError(
            f"Brysbaert norms not found at {NORMS_PATH}\n"
            "Make sure it's called concreteness_ratings.xlsx."
        )

    norms = load_norms(NORMS_PATH)
    con   = duckdb.connect(DB_PATH)

    df = run_stability(con, norms)
    con.close()

    out = os.path.join(RESULTS_DIR, "sample_stability.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows → {out}")

    print_summary(df)
