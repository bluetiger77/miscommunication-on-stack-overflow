"""
Stream the .7z archive and write Posts.xml, Comments.xml, Tags.xml
to Parquet files in data/parquet/.
"""

import sys
import os
import xml.etree.ElementTree as ET
import pyarrow as pa
import pyarrow.parquet as pq
import py7zr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Walk up two directory levels from this script's location (src/ → project root).
# Using __file__ rather than os.getcwd() makes the path stable regardless of
# what working directory the script is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The raw Stack Overflow data dump; should be at the project root.
ARCHIVE = os.path.join(ROOT, "stackoverflow.com.7z")

# Folder for XML files extracted from the archive.
# These are large text files and are only needed long enough to stream into Parquet. Won't be needed after that.
RAW_DIR = os.path.join(ROOT, "data", "raw")

# Destination for the final Parquet files consumed by downstream pipeline stages.
PARQUET_DIR = os.path.join(ROOT, "data", "parquet")

# Create both directories
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)

# Only these three files are needed from the archive; everything else is skipped.
TARGET_FILES = {"Posts.xml", "Comments.xml", "Tags.xml"}

# Number of XML rows accumulated in memory before flushing a Parquet record batch.
# Larger values reduce I/O overhead; smaller values reduce peak memory usage.
BATCH_SIZE = 100_000

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

# Each schema is a list of (column_name, pyarrow_type) pairs. Declaring types
# explicitly here means PyArrow enforces them at write time rather than inferring
# them later, which catches bad data early and produces tighter Parquet files.

POSTS_FIELDS = [
    ("Id",           pa.int64()),   # Unique post identifier; use int64 in case ids are long
    ("PostTypeId",   pa.int8()),    # 1 = question, 2 = answer (small integer, int8 is sufficient)
    ("ParentId",     pa.int64()),   # For answers: the Id of the parent question; NULL for questions
    ("Body",         pa.large_utf8()),  # HTML body; large_utf8 supports strings greater than 2 gb
    ("Title",        pa.utf8()),    # Question title; only present on PostTypeId == 1
    ("Tags",         pa.utf8()),    # Raw tag string from the XML, e.g. "<python><pandas>"
    ("CreationDate", pa.utf8()),    # ISO 8601 timestamp; no timezone info is included by stack overflow
    ("Score",        pa.int32()),   # Net upvote/downvote score; can be negative
    ("AnswerCount",  pa.int32()),   # Number of answers posted under this question
    ("ViewCount",    pa.int32()),   # Total page views
]

COMMENTS_FIELDS = [
    ("Id",           pa.int64()),   # Unique comment identifier
    ("PostId",       pa.int64()),   # The post this comment is attached to
    ("Text",         pa.utf8()),    # Plain-text comment body (no HTML)
    ("CreationDate", pa.utf8()),    # ISO 8601 timestamp
    ("Score",        pa.int32()),   # Net upvote count on the comment
]

TAGS_FIELDS = [
    ("Id",        pa.int32()),      # Unique tag identifier
    ("TagName",   pa.utf8()),       # The tag string, e.g. "python"
    ("Count",     pa.int32()),      # Number of questions carrying this tag
]

# Maps each XML filename to its list of fields and the name to be used for the output file.
# The output name becomes the stem of the .parquet filename (e.g. "posts" → posts.parquet).
SCHEMA_MAP = {
    "Posts.xml":    (POSTS_FIELDS,    "posts"),
    "Comments.xml": (COMMENTS_FIELDS, "comments"),
    "Tags.xml":     (TAGS_FIELDS,     "tags"),
}

def _convertToInt(value, default=None):
    # All XML attribute values arrive as strings (or None if missing).
    # This helper converts to int safely, returning `default` rather than
    # raising an exception, which keeps the pipeline running past malformed rows.
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def row_from_elemement(filename, element):
    # element.attrib is a plain dictionary structured like {attribute_name: string_value} for the
    # each <row> element.
    attribute = element.attrib

    if filename == "Posts.xml":
        return (
            _convertToInt(attribute.get("Id")),              # NULL if missing; downstream can decide how to handle
            _convertToInt(attribute.get("PostTypeId"), 0),   # Default 0 signals an unknown post type
            _convertToInt(attribute.get("ParentId")),        # NULL for questions (no parent); int for answers
            attribute.get("Body"),                  # Raw HTML; kept as-is for later cleaning stages
            attribute.get("Title"),                 # NULL on answer rows
            attribute.get("Tags"),                  # NULL on answer rows; e.g. "<python><pandas>"
            attribute.get("CreationDate"),          # e.g. "2023-01-15T10:30:00.000"
            _convertToInt(attribute.get("Score"), 0),
            _convertToInt(attribute.get("AnswerCount"), 0),
            _convertToInt(attribute.get("ViewCount"), 0),
        )
    elif filename == "Comments.xml":
        return (
            _convertToInt(attribute.get("Id")),
            _convertToInt(attribute.get("PostId")),
            attribute.get("Text"),
            attribute.get("CreationDate"),
            _convertToInt(attribute.get("Score"), 0),
        )
    elif filename == "Tags.xml":
        return (
            _convertToInt(attribute.get("Id")),
            attribute.get("TagName"),
            _convertToInt(attribute.get("Count"), 0),
        )

    # filename not in SCHEMA_MAP — should never happen given how callers filter,
    # but returning None lets the caller skip the row gracefully.
    return None

def stream_xml_to_parquet(xml_path, filename):
    # Look up the schema and derive the output path from the table name.
    fields, table_name = SCHEMA_MAP[filename]
    schema = pa.schema(fields)
    out_path = os.path.join(PARQUET_DIR, f"{table_name}.parquet")

    # Open the Parquet writer once for the whole file. Snappy compression is
    # fast; it compresses well enough for columnar text data
    # without becoming a CPU bottleneck when writing to the parquet file.
    writer = pq.ParquetWriter(out_path, schema, compression="snappy")

    # Rows collect here until there are enough to write to the file in one go.
    batch_rows = []
    total = 0  # Running count of rows written, used only for progress reporting.

    # Reads the XML one element at a time instead of loading the whole file.
    # ("end",) means each element is processed after its closing tag, so all
    # its attributes are available.
    context = ET.iterparse(xml_path, events=("end",))
    for event, element in context:
        # The Stack Overflow XML schema wraps all data in <row> elements.
        # The root element (e.g. <posts>) and any other wrapper tags are skipped.
        if element.tag != "row":
            # Clear non-row elements immediately; they hold no useful data
            # and would accumulate in memory if left in the tree.
            element.clear()
            continue

        row = row_from_elemement(filename, element)

        # Critical: clear the element right after extracting its attributes.
        # ElementTree builds a tree in memory by default; clearing the element
        # after use lets the garbage collector reclaim it, keeping memory usage flat
        # even for XML files that are several gigabytes.
        element.clear()

        if row is None:
            # row_from_element returns None for unrecognized file types; skip.
            continue

        batch_rows.append(row)

        if len(batch_rows) >= BATCH_SIZE:
            # Rearrange from a list of rows into a list of columns, one per field.
            cols = list(zip(*batch_rows))

            # Wrap each column in a typed PyArrow array. The explicit `type=`
            # argument causes a cast/validation at this point rather than at
            # query time, so type errors surface here close to the data.
            arrays = [pa.array(c, type=fields[i][1]) for i, c in enumerate(cols)]

            # Write one record batch to the Parquet file. Parquet's columnar
            # layout means each batch appends column stripes sequentially,
            # which is efficient for large analytical reads.
            writer.write_batch(pa.record_batch(arrays, schema=schema))

            total += len(batch_rows)

            # \r overwrites the same terminal line on each flush, giving a live
            # counter without flooding stdout with thousands of progress lines.
            print(f"  [{filename}] written {total:,} rows", end="\r", flush=True)

            # Reset the buffer
            batch_rows.clear()

    # Finish processing the final partial batch. This handles the common case where the
    # total row count is not a perfect multiple of BATCH_SIZE.
    if batch_rows:
        cols = list(zip(*batch_rows))
        arrays = [pa.array(c, type=fields[i][1]) for i, c in enumerate(cols)]
        writer.write_batch(pa.record_batch(arrays, schema=schema))
        total += len(batch_rows)

    # Finalize the Parquet file. ParquetWriter buffers metadata (row group
    # offsets, statistics, schema) and writes the footer only on close().
    # Forgetting this call would produce a corrupt file.
    writer.close()

    # Print progress report
    print(f"  [{filename}] done — {total:,} rows → {out_path}")

def extract_and_convert():
    print(f"Opening archive: {ARCHIVE}")
    with py7zr.SevenZipFile(ARCHIVE, mode="r") as archive:
        # getnames() lists all paths inside the archive without decompressing them.
        names = archive.getnames()

        # Filter to only the files we need. os.path.basename handles cases where
        # the archive stores files under a subdirectory (e.g. "dump/Posts.xml").
        targets = [n for n in names if os.path.basename(n) in TARGET_FILES]
        print(f"Files to extract: {targets}")

        # Extract only the targeted files. Extracting the full archive would
        # decompress gigabytes of data (Badges, Users, etc.) that this pipeline
        # does not use.
        archive.extract(path=RAW_DIR, targets=targets)

    # After extraction, locate each XML file by walking the raw directory.
    # py7zr may place files directly in RAW_DIR or inside a named subfolder
    # depending on how the archive was created, so os.walk handles both layouts.
    for filename in TARGET_FILES:
        xml_path = None
        for root, _, files in os.walk(RAW_DIR):
            if filename in files:
                xml_path = os.path.join(root, filename)
                break  # Stop at the first match; there should only ever be one.

        if xml_path is None:
            # Non-fatal: warn and continue so the rest of the pipeline can proceed
            # with whatever files were found.
            print(f"WARNING: {filename} not found after extraction, skipping.")
            continue

        print(f"Parsing {xml_path} ...")
        stream_xml_to_parquet(xml_path, filename)

if __name__ == "__main__":
    extract_and_convert()
    print("Phase 1 complete.")
