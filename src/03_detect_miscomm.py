"""
Scan all answers and comments for phrases that signal miscommunication (using regex patterns).
Patterns are organized into semantic categories. Results are saved per answer and per
comment with one boolean column per category, then combined into a single row per question
with per-category flags (True if any answer or comment in that question matched).

Outputs:
  - data/parquet/answers_flagged.parquet  (answer_id, question_id, has_miscomm, cat_*)
  - data/parquet/comments_flagged.parquet (comment_id, question_id, has_miscomm, cat_*)
  - question_miscomm table in the so.duckdb database (question_id, has_miscomm, cat_*)
"""

import os
import re
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from html import unescape

# Resolve project root so all paths are absolute, enabling launching from any working directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH     = os.path.join(ROOT, "data", "db", "so.duckdb")
PARQUET_DIR = os.path.join(ROOT, "data", "parquet")

# How many rows to fetch per loop iteration. Large enough to be efficient;
# small enough to fit in memory.
BATCH_SIZE  = 200_000

# ---------------------------------------------------------------------------
# Patterns — organized into 13 semantic categories
# ---------------------------------------------------------------------------
# Define a more realistic "word"
WORD = r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*"

# Maximum characters to match between keywords in a pattern. Caps matches to
# avoid crossing sentence boundaries.
MAX_PHRASE_LEN = 80

# CATEGORIES maps each ambiguity type to a single compiled pattern (alternatives
# joined with |). One re.search() call per category per text instead of N calls.
# \b matches word boundaries to avoid partial-word false matches.
# re.IGNORECASE makes matching case-insensitive throughout.
CATEGORIES: dict[str, re.Pattern] = {

    # -----------------------------------------------------------------------
    # General evidence of miscommunication — broad signals that something
    # wasn't understood, including clarification requests, partial understanding,
    # conditional reinterpretation, self-repair, explicit misunderstanding
    # acknowledgment, and resolution markers. These do not specify *what type*
    # of ambiguity caused the breakdown.
    # -----------------------------------------------------------------------
    "comprehension_failure": re.compile(
        r"\byou mean\b"
        r"|\bi don['']t understand\b"
        r"|\bi['']m not sure if you\b"
        r"|\bi['']m not sure what you\b"
        r"|\bi['']m confused by what you\b"
        r"|\bi can['']t tell if\b"
        rf"|\bwhat do you mean by\s+(.{{1,{MAX_PHRASE_LEN}}}?)\b"
        r"|\bi (kind of|sort of) (get|understand)\b"
        r"|\bthis is (a bit|kind of|somewhat) confusing\b"
        r"|\bi['']m not totally sure\b"
        r"|\blet me rephrase\b"
        r"|\bwhat i meant (was|is)\b"
        r"|\bthat came out wrong\b"
        r"|\bi should have said\b"
        r"|\bbut that['']s what i['']m asking\b"
        r"|\bi misunderstood\b"
        r"|\bi (took|interpreted) that as\b"
        r"|\bi read that as\b"
        r"|\bi thought you meant\b"
        rf"|\bif by\s+(.{{1,{MAX_PHRASE_LEN}}}?)\s+you mean\b"
        r"|\bif you mean\b"
        r"|\bdo you mean\b"
        r"|\bare you (saying|meaning)\b"
        r"|\bso you['']re saying\b"
        r"|\bi see now\b"
        r"|\bi understand now\b"
        r"|\bthat makes sense now\b"
        r"|\bokay i get it now\b"
        r"|\bi finally see\b"
        r"|\bi finally understand\b"
        r"|\bthat finally makes sense\b"
        r"|\bokay i finally get it\b"
        r"|\bthat clears it up\b",
        re.IGNORECASE,
    ),

    # -----------------------------------------------------------------------
    # Intent confusion — the reader doesn't know what the writer is trying
    # to do or accomplish (goal/purpose ambiguity).
    # -----------------------------------------------------------------------
    "intent_confusion": re.compile(
        rf"\bwhat is\s+(.{{1,{MAX_PHRASE_LEN}}}?)\s+trying to do\b"
        rf"|\bwhat is\s+({WORD})\s+(?:for|used for)\b"
        r"|\bwhat are you trying to (do|say)\b"
        rf"|\bwhat(?: is|['']s) the (point|reason|reasoning|purpose)\s+(.{{1,{MAX_PHRASE_LEN}}}?)\b"
        r"|\bi thought that['']s what you\b",
        re.IGNORECASE,
    ),

    # -----------------------------------------------------------------------
    # Terminology confusion — a specific word or identifier is not understood.
    # -----------------------------------------------------------------------
    "terminology": re.compile(
        rf"\bwhat does\s+({WORD})\s+mean\b"
        rf"|\bwhat is\s+({WORD})\s+supposed to mean\b"
        rf"|\bi['']m confused as to what\s+({WORD})\s+means\b"
        rf"|\bwhat is meant by\s+({WORD})\b"
        rf"|\bwhat do you mean by\s+({WORD})\b",
        re.IGNORECASE,
    ),

    # -----------------------------------------------------------------------
    # Referential ambiguity — it's unclear which object, part, or entity
    # is being referred to.
    # -----------------------------------------------------------------------
    "referential": re.compile(
        r"\bwhich (one|part|thing)\b"
        r"|\bwhat are you referring to\b"
        rf"|\bwhat does\s+({WORD})\s+refer to\b"
        rf"|\bwhat is\s+({WORD})\s+referring to\b"
        r"|\bwhat (is|are) that\b",
        re.IGNORECASE,
    ),

    # -----------------------------------------------------------------------
    # Missing information / underspecification — the writer left out something
    # needed to understand or act on the post.
    # -----------------------------------------------------------------------
    "missing_info": re.compile(
        r"\byou (didn['']t|did not) (say|mention|include)\b"
        r"|\byou forgot to (say|explain|specify|state|write|mention|show)\b"
        r"|\byou left out\b"
        r"|\bthere['']s no (info|information|details|detail)\b"
        r"|\byou (haven['']t|have not) (shown|provided|included|explained|mentioned)\b"
        r"|\bcan you (show|provide|include|share|post) (the|your)\b"
        r"|\bwhat (error|output|result|version) (are|do|did) you\b"
        r"|\bwhere (is|are) (the|your) (error|output|code|definition|declaration)\b",
        re.IGNORECASE,
    ),

    # -----------------------------------------------------------------------
    # Contradiction / inconsistency — the writer said something that conflicts
    # with what they said earlier.
    # -----------------------------------------------------------------------
    "contradiction": re.compile(
        r"\bbut (earlier|before|above|previously) you (said|wrote|mentioned)\b"
        r"|\bbut how does that relate to\b"
        r"|\bisn['']t that not what you want\b"
        r"|\bbut you just said\b"
        r"|\bthat['']s not what you (just )?said\b"
        r"|\bwait[,]? (didn['']t|did not) you (just )?say\b"
        r"|\byou['']re contradicting\b"
        r"|\bthat contradicts\b"
        r"|\bthat['']s the opposite of what you\b"
        r"|\bthat conflicts with (what you|your)\b",
        re.IGNORECASE,
    ),

    # -----------------------------------------------------------------------
    # Pragmatic / tone confusion — uncertainty about the speaker's intent
    # (sarcasm, irony, seriousness).
    # -----------------------------------------------------------------------
    "pragmatic_tone": re.compile(
        r"\bi can['']t tell if you['']re (serious|joking)\b"
        r"|\bwas that (sarcasm|a joke)\b"
        r"|\bare you (being )?serious\b"
        r"|\byou (can['']t|cannot) be serious\b"
        r"|\bare you joking\b"
        r"|\bi['']m (being |totally |completely )?serious\b"
        r"|\bthis is (a genuine|an honest) question\b"
        r"|\bi['']m not (trying to be |being )?sarcastic\b",
        re.IGNORECASE,
    ),

    # -----------------------------------------------------------------------
    # Actionability confusion — the reader doesn't know what action to take
    # or how to perform it.
    # -----------------------------------------------------------------------
    "actionability": re.compile(
        r"\bwhat (should|am i supposed to) do\b"
        r"|\bhow (do|am i supposed to|would i)\b",
        re.IGNORECASE,
    ),

    # -----------------------------------------------------------------------
    # Scope ambiguity — unclear which cases, items, or contexts something
    # applies to (applies to all? just some? a specific subset?).
    # -----------------------------------------------------------------------
    "scope_ambiguity": re.compile(
        r"\bdoes this apply to\b"
        r"|\bare we talking about\b"
        r"|\bis this for\b"
        r"|\bdo you mean (?:all|some|any|one|now|generally|in general)\b"
        r"|\bdoes this apply to (?:all|everything|every case)\b"
        r"|\bare you talking about (?:all|every case|a specific case)\b"
        r"|\bwhich ones exactly\b"
        r"|\bin what (context|scenario|case|situation)\b"
        r"|\bunder what (conditions|circumstances)\b"
        r"|\bis this (always|only|sometimes) (true|the case|applicable)\b"
        r"|\bfor (all|every|any|which) (cases?|scenarios?)\b",
        re.IGNORECASE,
    ),

    # -----------------------------------------------------------------------
    # Quantifier & distribution ambiguity — confusion about whether something
    # applies collectively vs. individually, inclusive/exclusive ranges,
    # universal vs. existential quantification, or per-item vs. aggregate.
    # -----------------------------------------------------------------------
    "quantifier_distribution": re.compile(
        r"\bdo(?:es)? (?:you|this|that|it) mean (?:each|every|all|both|any|one|a specific one)\b"
        r"|\bare you saying (?:each|every|all|both|any|one|a specific one)\b"
        r"|\bdoes (?:each|every) .{1,80}? (?:have|get|need|return|contain|use)\b"
        r"|\bis (?:this|that|it) (?:shared|common to all|per item|per element|inclusive|exclusive)\b"
        r"|\bdoes (?:this|that|it) include .{1,80}?\b"
        r"|\bdoes this include the end\b"
        r"|\bdoes this mean (?:every|all|any|only one|a specific)\b"
        r"|\bshould (?:all|every|any|only one|a specific) .{1,80}? (?:be|have|get|return|contain|use|match)\b"
        r"|\bdo(?:es)? (?:you|this|that|it) mean .{1,80}? (?:per item|per element|per row|per object|overall|total|collectively)\b"
        r"|\bis (?:this|that|it) .{1,80}? (?:per item|per element|per row|per object|overall|total|collectively)\b"
        r"|\b(?:per item|per element|per row|per object) or (?:overall|total|collectively)\b"
        r"|\b(?:overall|total|collectively) or (?:per item|per element|per row|per object)\b",
        re.IGNORECASE,
    ),

    # -----------------------------------------------------------------------
    # Logical structure ambiguity — unclear how conditions or clauses are
    # grouped (AND/OR scope, operator precedence, what applies to what).
    # -----------------------------------------------------------------------
    "logical_structure": re.compile(
        r"\bdo(?:es)? (?:you|this|that|it) mean both .{1,80}? and .{1,80}?\b"
        r"|\bare you saying .{1,80}? applies to both\b"
        r"|\bhow is this grouped\b"
        r"|\bwhat applies to what\b"
        r"|\bwhich has (priority|precedence)\b"
        r"|\bdoes (this|that|it) apply to (both|all|each)\b"
        r"|\bhow (do|does) (these|this) (group|nest|combine|associate)\b"
        rf"|\bwhich {WORD} (takes?|ha(?:s|ve)) (priority|precedence|effect)\b"
        r"|\bis (?:this|that|it) (?:and|or) or (?:and|or)\b"
        r"|\bwhat['']s the (grouping|nesting|evaluation order)\b",
        re.IGNORECASE,
    ),

    # -----------------------------------------------------------------------
    # Epistemic ambiguity — unclear whether something is known, assumed,
    # guaranteed, or merely a hypothesis.
    # -----------------------------------------------------------------------
    "epistemic": re.compile(
        r"\bdo we know that\b"
        r"|\bare we assuming\b"
        r"|\bare you assuming\b"
        r"|\bis this guaranteed\b"
        r"|\bdo you mean if it is known to be\b"
        r"|\bis it (assumed|known)\b"
        r"|\bcan (we|you|i) (safely )?assume\b"
        r"|\bis it safe to assume\b"
        r"|\bhow (do|did) you know\b"
        r"|\bwhat makes you (think|say|believe|assume)\b"
        r"|\bare you sure (that|about|of|this)\b"
        r"|\bis (that|this|it) (always|necessarily|definitely) (true|the case|correct)\b"
        r"|\bcan (we|you|i) be sure\b",
        re.IGNORECASE,
    ),

    # -----------------------------------------------------------------------
    # Comparison & measurement ambiguity — unclear what scale, dimension,
    # or unit is being used for a comparison or measurement.
    # -----------------------------------------------------------------------
    "comparison_measurement": re.compile(
        r"\bthan what\b"
        r"|\bcompared to what\b"
        r"|\brelative to what\b"
        r"|\bmore than what\b"
        r"|\bwhat units is\b"
        r"|\bin what units\b",
        re.IGNORECASE,
    ),
}

CATEGORY_NAMES = list(CATEGORIES.keys())
ALL_PATTERNS   = list(CATEGORIES.values())

# Possible additional categories:

# # Missing argument / role ambiguity (underspecified roles like input)
# re.compile(rf"\bwhat is\s+({WORD})\s+applied to\b", re.IGNORECASE),
# re.compile(rf"\bwhat does\s+({WORD})\s+operate on\b", re.IGNORECASE),
# re.compile(rf"\bwhat (?:does|is)\s+({WORD})\s+(?:depend|depending) on\b", re.IGNORECASE),
# re.compile(r"\bwhat is the input\b", re.IGNORECASE),

# # Identity / naming ambiguity
# re.compile(r"\bwhich (?:variable|object|class|method|function|property|field|table|column)\b", re.IGNORECASE),
# re.compile(r"\bwhat is\s+({WORD})\s+defined as\b", re.IGNORECASE),
# re.compile(r"\bwhere is\s+({WORD})\s+defined\b", re.IGNORECASE),
# re.compile(r"\bis\s+({WORD})\s+(?:a|an) (?:variable|function|method|class|object|type|string|array|list)\b", re.IGNORECASE),

# # Expected vs actual behavior ambiguity
# re.compile(r"\bwhat did you expect\b", re.IGNORECASE),
# re.compile(r"\bwhat is the expected (?:output|result|behavior|behaviour)\b", re.IGNORECASE),
# re.compile(r"\bwhat actually happens\b", re.IGNORECASE),
# re.compile(r"\bwhat output (?:do you get|are you getting)\b", re.IGNORECASE),
# re.compile(r"\bwhat error (?:do you get|are you getting)\b", re.IGNORECASE),

# # Unit/scale
# re.compile(r"\bseconds or milliseconds\b", re.IGNORECASE),
# re.compile(r"\bwhat scale\b", re.IGNORECASE),

def html_to_text(html: str) -> str:
    # Stack Overflow stores post bodies as HTML. Strip markup before
    # running regex patterns on the prose text.

    # Step 1: unescape HTML entities (&amp; → &, &lt; → <, etc.)
    text = unescape(html or "")

    # Step 2: Remove fenced code blocks (<pre><code>...</code></pre>).
    # Code blocks can contain identifiers that look like natural language
    # (e.g. iDontUnderstand), so drop them entirely. (?is) makes . match
    # newlines and disables case sensitivity.
    text = re.sub(r"(?is)<pre><code>.*?</code></pre>", " ", text)

    # Step 3: Remove inline code spans (<code>...</code>).
    # Same rationale — inline identifiers can look like natural language.
    text = re.sub(r"(?is)<code>.*?</code>",             " ", text)

    # Step 4: Strip all remaining HTML tags, leaving only prose text.
    text = re.sub(r"(?is)<.*?>",                        " ", text)

    # Step 5: Collapse runs of whitespace into a single space so patterns
    # with \s+ work predictably.
    text = re.sub(r"\s+", " ", text).strip()
    return text

def contains_miscomm(text: str) -> bool:
    # Returns True as soon as any pattern across all categories matches.
    return any(p.search(text) for p in ALL_PATTERNS)

def detect_categories(text: str) -> dict[str, bool]:
    # Returns one bool per category: True if the category's pattern matched.
    return {cat: bool(pattern.search(text))
            for cat, pattern in CATEGORIES.items()}

# ---------------------------------------------------------------------------
# Batch processing helpers
# ---------------------------------------------------------------------------

def _build_schema(id_field: str) -> pa.Schema:
    """Build the parquet schema: id + question_id + has_miscomm + one bool per category.
    id_field is the name of the primary key column ('answer_id' or 'comment_id')."""
    fields = [
        (id_field,    pa.int64()),
        ("question_id", pa.int64()),
        ("has_miscomm", pa.bool_()),
    ]
    for cat in CATEGORY_NAMES:
        fields.append((f"cat_{cat}", pa.bool_()))
    return pa.schema(fields)

def flag_answers(con: duckdb.DuckDBPyConnection):
    out_path = os.path.join(PARQUET_DIR, "answers_flagged.parquet")
    schema   = _build_schema("answer_id")
    writer   = pq.ParquetWriter(out_path, schema, compression="snappy")

    total = con.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
    print(f"Scanning {total:,} answers ...")

    offset = 0
    processed = 0
    while True:
        rows = con.execute(f"""
            SELECT answer_id, question_id, body
            FROM answers
            ORDER BY answer_id
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """).fetchall()

        if not rows:
            break

        ids, qids, flags = [], [], []
        cat_flags = {cat: [] for cat in CATEGORY_NAMES}

        for answer_id, question_id, body in rows:
            text = html_to_text(body or "")
            if contains_miscomm(text):
                cats = detect_categories(text)
            else:
                cats = {cat: False for cat in CATEGORY_NAMES}
            ids.append(answer_id)
            qids.append(question_id)
            flags.append(any(cats.values()))
            for cat in CATEGORY_NAMES:
                cat_flags[cat].append(cats[cat])

        arrays = [
            pa.array(ids,   pa.int64()),
            pa.array(qids,  pa.int64()),
            pa.array(flags, pa.bool_()),
        ] + [pa.array(cat_flags[cat], pa.bool_()) for cat in CATEGORY_NAMES]

        writer.write_batch(pa.record_batch(arrays, schema=schema))
        offset    += BATCH_SIZE
        processed += len(rows)
        print(f"  answers: {processed:,}/{total:,}", end="\r", flush=True)

    writer.close()
    print(f"\n  → {out_path}")
    return out_path

def flag_comments(con: duckdb.DuckDBPyConnection):
    out_path = os.path.join(PARQUET_DIR, "comments_flagged.parquet")
    schema   = _build_schema("comment_id")
    writer   = pq.ParquetWriter(out_path, schema, compression="snappy")

    total = con.execute("SELECT COUNT(*) FROM comments_enriched WHERE question_id IS NOT NULL").fetchone()[0]
    print(f"Scanning {total:,} comments ...")

    offset = 0
    processed = 0
    while True:
        rows = con.execute(f"""
            SELECT comment_id, question_id, text
            FROM comments_enriched
            WHERE question_id IS NOT NULL
            ORDER BY comment_id
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """).fetchall()
        if not rows:
            break

        ids, qids, flags = [], [], []
        cat_flags = {cat: [] for cat in CATEGORY_NAMES}

        for comment_id, question_id, text in rows:
            text = text or ""
            if contains_miscomm(text):
                cats = detect_categories(text)
            else:
                cats = {cat: False for cat in CATEGORY_NAMES}
            ids.append(comment_id)
            qids.append(question_id)
            flags.append(any(cats.values()))
            for cat in CATEGORY_NAMES:
                cat_flags[cat].append(cats[cat])

        arrays = [
            pa.array(ids,   pa.int64()),
            pa.array(qids,  pa.int64()),
            pa.array(flags, pa.bool_()),
        ] + [pa.array(cat_flags[cat], pa.bool_()) for cat in CATEGORY_NAMES]

        writer.write_batch(pa.record_batch(arrays, schema=schema))
        offset    += BATCH_SIZE
        processed += len(rows)
        print(f"  comments: {processed:,}/{total:,}", end="\r", flush=True)

    writer.close()
    print(f"\n  → {out_path}")
    return out_path

def build_question_miscomm_table(con: duckdb.DuckDBPyConnection):
    answers_path  = os.path.join(PARQUET_DIR, "answers_flagged.parquet")
    comments_path = os.path.join(PARQUET_DIR, "comments_flagged.parquet")

    # For each category, aggregate across all answers/comments per question:
    # BOOL_OR returns True if any row for that question flagged the category.
    cat_selects = ",\n            ".join(
        f"BOOL_OR(cat_{cat}) AS cat_{cat}" for cat in CATEGORY_NAMES
    )

    # A question is flagged (overall or per-category) if any of its answers or
    # comments matched a pattern in that category.
    # UNION ALL keeps all rows; UNION would incorrectly deduplicate.
    con.execute(f"""
        CREATE OR REPLACE TABLE question_miscomm AS
        SELECT
            question_id,
            BOOL_OR(has_miscomm) AS has_miscomm,
            {cat_selects}
        FROM (
            SELECT question_id, has_miscomm, {", ".join(f"cat_{c}" for c in CATEGORY_NAMES)}
            FROM read_parquet('{answers_path}')
            UNION ALL
            SELECT question_id, has_miscomm, {", ".join(f"cat_{c}" for c in CATEGORY_NAMES)}
            FROM read_parquet('{comments_path}')
        )
        GROUP BY question_id
    """)

    n_total   = con.execute("SELECT COUNT(*) FROM question_miscomm").fetchone()[0]
    n_flagged = con.execute("SELECT COUNT(*) FROM question_miscomm WHERE has_miscomm").fetchone()[0]
    print(f"question_miscomm: {n_total:,} questions, {n_flagged:,} flagged ({n_flagged/n_total*100:.2f}%)")

    # Print per-category counts as a quick sanity check.
    for cat in CATEGORY_NAMES:
        n = con.execute(f"SELECT COUNT(*) FROM question_miscomm WHERE cat_{cat}").fetchone()[0]
        print(f"  {cat}: {n:,} ({n/n_total*100:.2f}%)")

if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    flag_answers(con)
    flag_comments(con)
    build_question_miscomm_table(con)
    con.close()
    print("Phase 3 complete.")
