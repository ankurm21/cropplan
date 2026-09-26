"""
enrich_metadata.py
────────────────────────────────────────────────────────────
Gentle, hardware-friendly metadata enrichment pass.

This script reads all chunks already stored in ChromaDB,
sends each one (missing proper metadata) to Ollama for
topic/entity extraction, and updates the vector store
IN PLACE — no re-embedding needed.

Hardware-friendly design:
    • Processes in small batches  (default: 10 chunks)
    • Sleeps between batches       (default: 3 seconds)
    • Sleeps between LLM calls     (default: 1 second)
    • Saves progress in a ledger so you can stop/resume
    • Limits concurrent work to ~5% CPU baseline

Usage (from backend/ directory):
    python -m rag.services.enrich_metadata
    python -m rag.services.enrich_metadata --batch 5 --sleep 5
    python -m rag.services.enrich_metadata --resume
────────────────────────────────────────────────────────────
"""

import json
import sys
import time
import logging
import gc
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ENRICH_LEDGER = BACKEND_DIR / "rag" / "enrich_progress.json"

# ── Imports ───────────────────────────────────────────────
try:
    from .vector_store import get_collection
    from .chunk_derivation import llm_extract_metadata, normalize_primary_topic, clean_list
except ImportError:
    sys.path.insert(0, str(BACKEND_DIR))
    from rag.services.vector_store import get_collection
    from rag.services.chunk_derivation import llm_extract_metadata, normalize_primary_topic, clean_list


# ── Progress helpers ──────────────────────────────────────

def load_progress() -> set:
    if ENRICH_LEDGER.exists():
        data = json.loads(ENRICH_LEDGER.read_text(encoding="utf-8"))
        return set(data.get("done_ids", []))
    return set()


def save_progress(done_ids: set):
    ENRICH_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ENRICH_LEDGER.write_text(
        json.dumps({"done_ids": list(done_ids), "count": len(done_ids)}, indent=2),
        encoding="utf-8",
    )


# ── Main enrichment ──────────────────────────────────────

def enrich(
    batch_size: int = 10,
    sleep_between_batches: float = 3.0,
    sleep_between_calls: float = 1.0,
    max_chunks: int = 0,       # 0 = all
):
    """
    Walk through ChromaDB, find chunks with default "Research" topic,
    call Ollama to extract real metadata, and update in-place.
    """
    col = get_collection()
    total = col.count()
    logger.info(f"Collection has {total} chunks total")

    done_ids = load_progress()
    logger.info(f"Already enriched: {len(done_ids)} chunks (from previous runs)")

    # Fetch all IDs + metadatas to find un-enriched ones
    # ChromaDB get() with limit
    FETCH_BATCH = 500
    pending_ids = []
    pending_texts = []

    offset = 0
    while offset < total:
        batch = col.get(
            limit=FETCH_BATCH,
            offset=offset,
            include=["metadatas", "documents"],
        )

        if not batch["ids"]:
            break

        for i, uid in enumerate(batch["ids"]):
            if uid in done_ids:
                continue

            meta = batch["metadatas"][i]
            # Only enrich chunks that still have the default "Research" topic
            if meta.get("primary_topic", "Research") == "Research":
                pending_ids.append(uid)
                pending_texts.append(batch["documents"][i])

        offset += len(batch["ids"])

    logger.info(f"Chunks needing enrichment: {len(pending_ids)}")

    if max_chunks > 0:
        pending_ids = pending_ids[:max_chunks]
        pending_texts = pending_texts[:max_chunks]
        logger.info(f"Capped to first {max_chunks} chunks")

    enriched = 0
    failed = 0

    for i in range(0, len(pending_ids), batch_size):
        batch_ids = pending_ids[i : i + batch_size]
        batch_txts = pending_texts[i : i + batch_size]

        update_ids = []
        update_metas = []

        for j, (uid, text) in enumerate(zip(batch_ids, batch_txts)):
            try:
                meta = llm_extract_metadata(text[:1500])   # cap text length

                if meta:
                    new_meta = {
                        "primary_topic":     normalize_primary_topic(meta.get("primary_topic")),
                        "crop_entities":     ", ".join(clean_list(meta.get("crop_entities"))),
                        "practice_entities": ", ".join(clean_list(meta.get("practice_entities"))),
                        "region_entities":   ", ".join(clean_list(meta.get("region_entities"))),
                        "numeric_signals":   ", ".join(clean_list(meta.get("numeric_signals"))),
                        "relevance_score":   max(0.0, min(1.0, float(meta.get("relevance_score", 0.5)))),
                        "raw_text":          text[:1500],  # preserve original text for LLM prompts
                    }
                    update_ids.append(uid)
                    update_metas.append(new_meta)
                    enriched += 1
                else:
                    failed += 1

            except Exception as e:
                logger.warning(f"  LLM call failed for {uid}: {e}")
                failed += 1

            # ── Gentle sleep between individual LLM calls ──
            if sleep_between_calls > 0 and j < len(batch_ids) - 1:
                time.sleep(sleep_between_calls)

        # Batch update ChromaDB
        if update_ids:
            col.update(ids=update_ids, metadatas=update_metas)

        # Mark all as done (even failed — no point retrying broken text)
        done_ids.update(batch_ids)
        save_progress(done_ids)

        progress_pct = min(100, (i + len(batch_ids)) / len(pending_ids) * 100)
        logger.info(
            f"  Batch {i // batch_size + 1}: "
            f"{enriched} enriched, {failed} failed  "
            f"({progress_pct:.0f}% done)"
        )

        # ── Gentle sleep between batches — let CPU breathe ──
        if sleep_between_batches > 0 and i + batch_size < len(pending_ids):
            logger.info(f"  💤 Sleeping {sleep_between_batches}s to cool hardware…")
            gc.collect()     # free memory
            time.sleep(sleep_between_batches)

    logger.info("=" * 50)
    logger.info("ENRICHMENT COMPLETE")
    logger.info(f"  Enriched : {enriched}")
    logger.info(f"  Failed   : {failed}")
    logger.info(f"  Total done (cumulative): {len(done_ids)}")
    logger.info("=" * 50)


# ── CLI ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gently enrich chunk metadata via Ollama")
    parser.add_argument("--batch", type=int, default=10,
                        help="Chunks per batch (default: 10)")
    parser.add_argument("--sleep", type=float, default=3.0,
                        help="Seconds between batches (default: 3)")
    parser.add_argument("--call-sleep", type=float, default=1.0,
                        help="Seconds between individual LLM calls (default: 1)")
    parser.add_argument("--max", type=int, default=0,
                        help="Max chunks to enrich (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last saved progress (default behaviour)")
    args = parser.parse_args()

    enrich(
        batch_size=args.batch,
        sleep_between_batches=args.sleep,
        sleep_between_calls=args.call_sleep,
        max_chunks=args.max,
    )
