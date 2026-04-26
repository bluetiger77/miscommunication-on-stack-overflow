"""
Compute a concreteness/abstractness score for each topic (both tracks) using
the Brysbaert et al. (2014) concreteness norms.

Method:
  For each topic, randomly sample up to SAMPLE_SIZE question titles.
  Tokenize each title (lowercase, alphabetic words only).
  Look up each token in the Brysbaert lexicon
  Average matched scores across all tokens in all sampled titles.
  Higher score = more concrete; lower = more abstract.

Question titles are used rather than bodies or comments

Inputs:
  data/concreteness_ratings.xlsx  (Brysbaert et al. 2014 concreteness scores)
  data/db/so.duckdb               (question_tags, questions)

Outputs:
  results/topic_abstractness_tags.csv  (topic, n_sampled, n_words_matched, concreteness_score)
"""

import os
import re
import duckdb
import pandas as pd

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH     = os.path.join(ROOT, "data", "db", "so.duckdb")
NORMS_PATH  = os.path.join(ROOT, "data", "concreteness_ratings.xlsx")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Number of question titles to sample per topic. Large enough for a stable
# average; small enough to run quickly.
SAMPLE_SIZE = 500

# Minimum number of questions a topic must have to be included (same as the rest of the pipeline).
MIN_QUESTIONS = 1000

def load_norms(path: str) -> dict[str, float]:
    """Load Brysbaert norms into a lowercase word → concreteness score dict."""
    df = pd.read_excel(path, usecols=["Word", "Conc.M"])
    norms = {str(row["Word"]).lower(): float(row["Conc.M"])
             for _, row in df.iterrows()
             if pd.notna(row["Word"]) and pd.notna(row["Conc.M"])}
    print(f"Loaded {len(norms):,} words from Brysbaert norms.")
    return norms

def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphabetic words only (skip numbers, code terms)."""
    return re.findall(r"[a-z]+", (text or "").lower())

def score_titles(titles: list[str], norms: dict[str, float]) -> tuple[int, float | None]:
    """
    Score a list of titles against the norms.
    Returns (n_words_matched, mean_concreteness) or (0, None) if no words matched.
    """
    scores = []
    for title in titles:
        for token in tokenize(title):
            if token in norms:
                scores.append(norms[token])
    if not scores:
        return 0, None
    return len(scores), sum(scores) / len(scores)

def score_topics_tags(con: duckdb.DuckDBPyConnection, norms: dict[str, float]) -> pd.DataFrame:
    """Track A: score each tag by sampling question titles."""
    # Get all qualifying tags.
    tags = con.execute(f"""
        SELECT tag
        FROM question_tags
        GROUP BY tag
        HAVING COUNT(*) >= {MIN_QUESTIONS}
        ORDER BY tag
    """).fetchdf()["tag"].tolist()

    print(f"Scoring {len(tags)} tags ...")
    rows = []
    for i, tag in enumerate(tags, 1):
        titles = con.execute(f"""
            SELECT q.title
            FROM question_tags t
            JOIN questions q ON t.question_id = q.question_id
            WHERE t.tag = ?
              AND q.title IS NOT NULL
            ORDER BY random()
            LIMIT {SAMPLE_SIZE}
        """, [tag]).fetchdf()["title"].tolist()

        n_matched, score = score_titles(titles, norms)
        rows.append({
            "topic":               tag,
            "n_sampled":          len(titles),
            "n_words_matched":    n_matched,
            "concreteness_score": score,
        })
        print(f"  {i}/{len(tags)} {tag}: n={len(titles)}, score={f'{score:.3f}' if score else 'N/A'}",
              end="\r", flush=True)

    print()
    return pd.DataFrame(rows)

if __name__ == "__main__":
    if not os.path.exists(NORMS_PATH):
        raise FileNotFoundError(
            f"You need to put the ratings of concreteness scores at {NORMS_PATH}\n"
            "Make sure it's called concreteness_ratings.xlsx."
        )

    norms = load_norms(NORMS_PATH)
    con   = duckdb.connect(DB_PATH)

    tags_df = score_topics_tags(con, norms)
    out_tags = os.path.join(RESULTS_DIR, "topic_abstractness_tags.csv")
    tags_df.to_csv(out_tags, index=False)
    print(f"Tags abstractness: {len(tags_df)} rows → {out_tags}")
    print(f"  Score range: {tags_df['concreteness_score'].min():.3f} – "
          f"{tags_df['concreteness_score'].max():.3f}  "
          f"(mean {tags_df['concreteness_score'].mean():.3f})")

    con.close()
    print("Phase 8 complete.")
