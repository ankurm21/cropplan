"""
reembed_enriched.py
────────────────────────────────────────────────────────────
Re-embed all chunks with metadata-augmented documents.

After running enrich_metadata.py (which populates topic, crops,
practices, regions, numerics in metadata), this script:

1. Reads every chunk from ChromaDB
2. Builds enriched documents: [Topic: X] [Crops: Y] + raw text
3. Re-upserts so embeddings now encode metadata entities

This is a ONE-TIME operation after enrichment completes.
Run time: ~15-25 minutes for 3,581 chunks on CPU.

Usage (from backend/ directory, with venv activated):
    python -m rag.services.reembed_enriched
    python -m rag.services.reembed_enriched --batch 100
────────────────────────────────────────────────────────────
"""

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

try:
    from .vector_store import get_collection, build_enriched_document, _get_embed_fn
except ImportError:
    sys.path.insert(0, str(BACKEND_DIR))
    from rag.services.vector_store import get_collection, build_enriched_document, _get_embed_fn


def reembed(batch_size: int = 100):
    """
    Read all chunks, rebuild enriched documents from metadata,
    and re-upsert with new embeddings.
    """
    col = get_collection()
    total = col.count()
    logger.info(f"Collection has {total} chunks to re-embed")

    try:
        from .vector_store import NomicEmbeddingFunction
        embed_fn = NomicEmbeddingFunction(device="cuda", is_query=False)
    except Exception as e:
        logger.error(f"Failed to load GPU model: {e}. Falling back to CPU.")
        from .vector_store import NomicEmbeddingFunction
        embed_fn = NomicEmbeddingFunction(device="cpu", is_query=False)

    FETCH_BATCH = 500
    offset = 0
    processed = 0
    enriched_count = 0

    while offset < total:
        # Fetch a batch of chunks with their metadata and documents
        batch = col.get(
            limit=FETCH_BATCH,
            offset=offset,
            include=["metadatas", "documents"],
        )

        if not batch["ids"]:
            break

        # Process in smaller sub-batches for embedding
        batch_ids = batch["ids"]
        batch_metas = batch["metadatas"]
        batch_docs = batch["documents"]

        for i in range(0, len(batch_ids), batch_size):
            sub_ids = batch_ids[i:i + batch_size]
            sub_metas = batch_metas[i:i + batch_size]
            sub_docs = batch_docs[i:i + batch_size]

            new_documents = []
            new_metadatas = []

            for uid, meta, doc in zip(sub_ids, sub_metas, sub_docs):
                # Get raw text — prefer raw_text from metadata, else use document
                raw_text = meta.get("raw_text", "") or doc

                # Strip existing enrichment tags if document was already enriched
                # (tags start with [ and end before the actual text)
                if raw_text.startswith("[") and "\n" in raw_text:
                    # Already enriched document — extract raw text after first newline
                    raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text

                # Build new enriched document
                enriched = build_enriched_document(raw_text, meta)

                # Store raw_text in metadata if not already there
                if "raw_text" not in meta or not meta["raw_text"]:
                    meta["raw_text"] = raw_text

                new_documents.append(enriched)
                new_metadatas.append(meta)

                # Count how many actually got enrichment tags
                if enriched != raw_text:
                    enriched_count += 1

            # Generate new embeddings
            embeddings = embed_fn(new_documents)

            # Update in ChromaDB with new embeddings
            col.update(
                ids=sub_ids,
                documents=new_documents,
                metadatas=new_metadatas,
                embeddings=embeddings,
            )

            processed += len(sub_ids)
            pct = min(100, processed / total * 100)
            logger.info(
                f"  Re-embedded {processed}/{total} chunks "
                f"({pct:.0f}%) — {enriched_count} enriched so far"
            )

            # Brief pause between sub-batches
            time.sleep(0.3)
            gc.collect()

        offset += len(batch_ids)

    logger.info("=" * 50)
    logger.info("RE-EMBEDDING COMPLETE")
    logger.info(f"  Total processed:  {processed}")
    logger.info(f"  With enrichment:  {enriched_count}")
    logger.info(f"  Without (raw):    {processed - enriched_count}")
    logger.info("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Re-embed chunks with metadata-augmented documents")
    parser.add_argument("--batch", type=int, default=100,
                        help="Chunks per embedding batch (default: 100)")
    args = parser.parse_args()

    reembed(batch_size=args.batch)
