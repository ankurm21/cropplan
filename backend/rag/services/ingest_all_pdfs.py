"""
ingest_all_pdfs.py
────────────────────────────────────────────────────────────
Batch pipeline:   PDF  →  chunk_derivation  →  vector store

Usage (from backend/ directory):
    python -m rag.services.ingest_all_pdfs

What it does, per PDF:
    1.  Checks if the PDF has already been fully ingested
        (by looking for its doc_id in a local ledger file).
    2.  Runs stream_pdf_pages_true → preprocessing_1 → token bucketing.
    3.  Calls Ollama for metadata extraction on every bucket.
    4.  enforce_schema normalizes each chunk.
    5.  Upserts the chunk batch into ChromaDB via vector_store.upsert_chunks.
    6.  Marks the PDF as done in the ledger.

The ledger (rag/ingest_ledger.json) lets you re-run safely
without re-processing already ingested documents.
────────────────────────────────────────────────────────────
"""

import json
import os
import sys
import time
import gc
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Resolve paths ─────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent   # backend/
PDF_DIR     = BACKEND_DIR / "data" / "pdf"
LEDGER_PATH = BACKEND_DIR / "rag" / "ingest_ledger.json"

# ── Imports from sibling modules ──────────────────────────
try:
    from .utility import stream_pdf_pages_true, preprocessing_1, call_ollama
    from .chunk_derivation import (
        stream_token_buckets, llm_extract_metadata,
        enforce_schema, TARGET_TOKENS, tokenizer, HF_MODEL
    )
    from .vector_store import upsert_chunks, collection_stats
except ImportError:
    # Running as __main__
    sys.path.insert(0, str(BACKEND_DIR))
    from rag.services.utility import stream_pdf_pages_true, preprocessing_1, call_ollama
    from rag.services.chunk_derivation import (
        stream_token_buckets, llm_extract_metadata,
        enforce_schema, TARGET_TOKENS, tokenizer, HF_MODEL
    )
    from rag.services.vector_store import upsert_chunks, collection_stats


# ── Ledger helpers ────────────────────────────────────────

def load_ledger() -> dict:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    return {}


def save_ledger(ledger: dict):
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


# ── Core ingestion for one PDF ────────────────────────────

def ingest_one_pdf(pdf_path: Path, skip_llm: bool = False) -> list:
    """
    Process a single PDF → structured chunks.

    If skip_llm=True, metadata extraction is skipped
    (useful for fast testing without Ollama).
    """
    pdf_name = pdf_path.name
    logger.info(f"▶ Processing: {pdf_name}")
    t0 = time.time()

    # --- Stream pages → clean blocks → token buckets ---
    def cleaned_block_stream():
        for page_number, elements, source_name in stream_pdf_pages_true(str(pdf_path)):
            blocks = preprocessing_1(elements, source_name, page_number)
            for block in blocks:
                yield block

    all_chunks = []
    chunk_counter = 1

    for bucket in stream_token_buckets(cleaned_block_stream(), TARGET_TOKENS):
        text = bucket["text"].strip()
        if not text:
            continue

        # Hard-cap tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > TARGET_TOKENS:
            tokens = tokens[:TARGET_TOKENS]
            text = tokenizer.decode(tokens)

        # LLM metadata
        if skip_llm:
            meta = None
        else:
            try:
                meta = llm_extract_metadata(text)
            except Exception as e:
                logger.warning(f"  LLM metadata failed for chunk {chunk_counter}: {e}")
                meta = None

        structured = enforce_schema(meta, bucket, chunk_counter, text)
        all_chunks.append(structured)
        chunk_counter += 1

        if chunk_counter % 20 == 0:
            logger.info(f"  … {chunk_counter - 1} chunks so far")

    elapsed = time.time() - t0
    logger.info(f"  ✓ {len(all_chunks)} chunks extracted in {elapsed:.1f}s")
    return all_chunks


# ── Main batch runner ─────────────────────────────────────

def ingest_all(skip_llm: bool = False, force: bool = False, pdf_sleep: float = 5.0):
    """
    Walk every PDF in data/pdf/, chunk + embed + store.
    Hardware-friendly: sleeps between PDFs and runs gc.collect().
    """
    if not PDF_DIR.exists():
        logger.error(f"PDF directory not found: {PDF_DIR}")
        return

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    logger.info(f"Found {len(pdfs)} PDFs in {PDF_DIR}")

    ledger = load_ledger()
    total_new = 0

    for pdf_path in pdfs:
        doc_key = pdf_path.stem.replace(" ", "_")

        if doc_key in ledger and not force:
            logger.info(f"⏩ Skipping (already ingested): {pdf_path.name}")
            continue

        try:
            chunks = ingest_one_pdf(pdf_path, skip_llm=skip_llm)

            if chunks:
                n = upsert_chunks(chunks)
                total_new += n

                ledger[doc_key] = {
                    "file": pdf_path.name,
                    "chunks": len(chunks),
                    "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                save_ledger(ledger)

        except Exception as e:
            logger.error(f"✗ Failed on {pdf_path.name}: {e}", exc_info=True)
            continue
        finally:
            # ── Hardware-friendly: free memory & let CPU breathe ──
            gc.collect()
            if pdf_sleep > 0:
                logger.info(f"  💤 Sleeping {pdf_sleep}s to cool hardware…")
                time.sleep(pdf_sleep)

    # Summary
    stats = collection_stats()
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"  New chunks added : {total_new}")
    logger.info(f"  Total in store   : {stats['total_chunks']}")
    logger.info(f"  Documents        : {stats['distinct_documents']}")
    logger.info(f"  Topic breakdown  : {stats['topic_distribution']}")
    logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest all agriculture PDFs into ChromaDB")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip Ollama metadata extraction (fast mode)")
    parser.add_argument("--force", action="store_true",
                        help="Re-process all PDFs even if already in ledger")
    parser.add_argument("--single", type=str, default=None,
                        help="Process a single PDF file by name")
    parser.add_argument("--pdf-sleep", type=float, default=5.0,
                        help="Seconds to sleep between PDFs (default: 5)")
    args = parser.parse_args()

    if args.single:
        target = PDF_DIR / args.single
        if not target.exists():
            logger.error(f"File not found: {target}")
            sys.exit(1)
        chunks = ingest_one_pdf(target, skip_llm=args.skip_llm)
        if chunks:
            upsert_chunks(chunks)
            gc.collect()
            stats = collection_stats()
            logger.info(f"Done. Store: {stats['total_chunks']} chunks")
    else:
        ingest_all(skip_llm=args.skip_llm, force=args.force, pdf_sleep=args.pdf_sleep)
