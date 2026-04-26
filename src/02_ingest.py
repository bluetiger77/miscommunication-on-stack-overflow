"""
Load Parquet files into a database. Set up views and indexes so every
answer and comment can be traced back to its root question.

A view is a saved SQL query that behaves like a table — you can query it
like normal, but it doesn't store any data of its own. It just runs its
query fresh each time it's used.
"""

import os
import duckdb

# Resolve the project root relative to this script so the file works
# regardless of the working directory it is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The Parquet files produced by 01_extract.py.
PARQUET_DIR = os.path.join(ROOT, "data", "parquet")

# The database will live in a single file for future scripts to use
DB_PATH = os.path.join(ROOT, "data", "db", "so.duckdb")

# Ensure the db directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Absolute paths used inside SQL queries with read_parquet().
POSTS_PQ    = os.path.join(PARQUET_DIR, "posts.parquet")
COMMENTS_PQ = os.path.join(PARQUET_DIR, "comments.parquet")
TAGS_PQ     = os.path.join(PARQUET_DIR, "tags.parquet")

def build_db():
    connection = duckdb.connect(DB_PATH)

    # -----------------------------------------------------------------------
    # Load raw Parquet into persistent tables
    # -----------------------------------------------------------------------
    # CREATE OR REPLACE drops the old table first if it exists, making this
    # script safely re-runnable without needing to manually clear the database.

    print("Loading posts ...")
    connection.execute(f"""
        CREATE OR REPLACE TABLE posts AS
        SELECT * FROM read_parquet('{POSTS_PQ}')
    """)
    # posts holds both questions (PostTypeId=1) and answers (PostTypeId=2)
    # in a single table — the same structure as the raw Stack Overflow dump.

    print("Loading comments ...")
    connection.execute(f"""
        CREATE OR REPLACE TABLE comments AS
        SELECT * FROM read_parquet('{COMMENTS_PQ}')
    """)

    print("Loading tags ...")
    connection.execute(f"""
        CREATE OR REPLACE TABLE tags AS
        SELECT * FROM read_parquet('{TAGS_PQ}')
    """)

    # -----------------------------------------------------------------------
    # Indexes
    # -----------------------------------------------------------------------
    # Indexes speed up the JOIN and WHERE clauses used in later scripts.
    print("Creating indexes ...")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_posts_id         ON posts(Id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_posts_type       ON posts(PostTypeId)")
    # ParentId links answers to their question; used in the comments_enriched JOIN.
    connection.execute("CREATE INDEX IF NOT EXISTS idx_posts_parent     ON posts(ParentId)")
    # PostId in comments references posts.Id; indexed for the enriched JOIN.
    connection.execute("CREATE INDEX IF NOT EXISTS idx_comments_postid  ON comments(PostId)")

    # ------------------------------------------------------------------
    # Views: questions and answers
    # ------------------------------------------------------------------
    # These views split the posts table into questions and answers by
    # filtering on PostTypeId, and rename columns to cleaner names so
    # later scripts don't need to know the raw Stack Overflow field names.

    connection.execute("""
        CREATE OR REPLACE VIEW questions AS
        SELECT
            Id           AS question_id,
            Title        AS title,
            Tags         AS raw_tags,   -- raw "<python><pandas>" string; parsed later
            Body         AS body,
            CreationDate AS created_at,
            Score        AS score,
            AnswerCount  AS answer_count,
            ViewCount    AS view_count
        FROM posts
        WHERE PostTypeId = 1            -- 1 = question in the SO schema
    """)

    connection.execute("""
        CREATE OR REPLACE VIEW answers AS
        SELECT
            Id           AS answer_id,
            ParentId     AS question_id, -- ParentId is the question this answer belongs to
            Body         AS body,
            CreationDate AS created_at,
            Score        AS score
        FROM posts
        WHERE PostTypeId = 2             -- 2 = answer in the SO schema
    """)

    # ------------------------------------------------------------------
    # Comments enriched with question_id
    # Comments on a question  → question_id = PostId
    # Comments on an answer   → question_id = answer's ParentId
    # ------------------------------------------------------------------
    # Each comment records which post it was left on (PostId), but that post
    # may itself be an answer. This view looks one level up to find the root
    # question in that case.
    #
    # The CASE expression handles this:
    #   - If the comment's post is a question, PostId IS the question_id.
    #   - If the comment's post is an answer, the question_id is that answer's ParentId.
    #   - NULL is returned for any unexpected post type (defensive; should not occur).
    connection.execute("""
        CREATE OR REPLACE VIEW comments_enriched AS
        SELECT
            c.Id      AS comment_id,
            c.PostId  AS post_id,
            CASE
                WHEN p.PostTypeId = 1 THEN c.PostId   -- comment on a question
                WHEN p.PostTypeId = 2 THEN p.ParentId  -- comment on an answer → go up to question
                ELSE NULL
            END       AS question_id,
            c.Text    AS text,
            c.CreationDate AS created_at,
            c.Score   AS score
        FROM comments c
        JOIN posts p ON c.PostId = p.Id  -- join to learn the type of the commented-on post
    """)

    # Check row counts before closing, so any extraction issue is visible immediately.
    n_q = connection.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    n_a = connection.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
    n_c = connection.execute("SELECT COUNT(*) FROM comments_enriched").fetchone()[0]
    print(f"questions: {n_q:,}  answers: {n_a:,}  comments: {n_c:,}")

    connection.close()
    print(f"Database written to {DB_PATH}")

if __name__ == "__main__":
    build_db()
    print("Phase 2 complete.")
