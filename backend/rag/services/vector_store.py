"""
vector_store.py
────────────────────────────────────────────────────────────
Persistent ChromaDB-backed vector store for KrishiPlan RAG.

Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings
(free, CPU-friendly, 384-dim, excellent semantic quality).

Collection schema per document:
    id          – unique composite key  (doc_id::chunk_id)
    document    – the chunk text
    embedding   – 384-d float vector  (auto-generated)
    metadata    – rich agriculture metadata dict
────────────────────────────────────────────────────────────
"""

import os
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

# Suppress annoying HuggingFace / Transformers warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Set external library loggers to ERROR to prevent console spam
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent          # backend/
CHROMA_PERSIST_DIR = str(BASE_DIR / "rag" / "chroma_db")

# ── Embedding Model ───────────────────────────────────────
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"      # 768-dim, semantic search
COLLECTION_NAME = "krishiplan_agri_docs_nomic"

class NomicEmbeddingFunction(EmbeddingFunction):
    def __init__(self, device: str = "cpu", is_query: bool = False):
        logger.info(f"Loading {EMBEDDING_MODEL} on {device}")
        from sentence_transformers import SentenceTransformer
        # nomic requires trust_remote_code=True
        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
            device=device
        )
        self.prefix = "search_query: " if is_query else "search_document: "
        
    def __call__(self, input: Documents) -> Embeddings:
        prefixed_input = [self.prefix + doc for doc in input]
        return self.model.encode(prefixed_input).tolist()

# ── Singleton ─────────────────────────────────────────────

_client: Optional[chromadb.ClientAPI] = None
_collection = None
_embed_fn = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        # Default for the collection is CPU, configured for search queries
        _embed_fn = NomicEmbeddingFunction(device="cpu", is_query=True)
    return _embed_fn


def get_collection():
    """Return (or create) the persistent ChromaDB collection."""
    global _client, _collection

    if _collection is not None:
        return _collection

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embed_fn(),
        metadata={"hnsw:space": "cosine"},      # cosine similarity
    )

    logger.info(
        f"ChromaDB collection '{COLLECTION_NAME}' ready  "
        f"({_collection.count()} docs)  persist={CHROMA_PERSIST_DIR}"
    )
    return _collection


# ── Public API ────────────────────────────────────────────

def build_enriched_document(text: str, meta: Dict[str, Any]) -> str:
    """
    Prepend metadata tags to chunk text for richer embeddings.

    The embedding model encodes this COMBINED string, so semantic
    search will match on entity names, topics, and regions —
    not just the raw paragraph text.

    Example output:
        [Topic: Advisory] [Crops: rice, wheat] [Practices: drip irrigation]
        [Region: Maharashtra] [Signals: pH 6.5, yield 40 q/ha]
        Actual chunk text here...
    """
    tags = []

    topic = meta.get("primary_topic", "")
    if topic and topic != "Research":  # skip default/generic topic
        tags.append(f"[Topic: {topic}]")

    crops = meta.get("crop_entities", "")
    if crops:
        tags.append(f"[Crops: {crops}]")

    practices = meta.get("practice_entities", "")
    if practices:
        tags.append(f"[Practices: {practices}]")

    regions = meta.get("region_entities", "")
    if regions:
        tags.append(f"[Region: {regions}]")

    signals = meta.get("numeric_signals", "")
    if signals:
        tags.append(f"[Data: {signals}]")

    if tags:
        prefix = " ".join(tags) + "\n"
        return prefix + text

    return text  # no enrichment available, embed raw text

def upsert_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Insert or update a batch of structured chunks into the vector store.

    Each chunk dict is expected to have AT LEAST:
        text, doc_id, chunk_id

    Optional (enriched metadata):
        document_name, page_numbers, primary_topic,
        crop_entities, practice_entities, region_entities,
        numeric_signals, relevance_score

    The 'documents' field sent to ChromaDB is the ENRICHED text
    (metadata tags + raw text) for better embeddings.
    The raw text is preserved in metadata['raw_text'] for LLM prompts.
    """
    col = get_collection()

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        uid = f"{chunk['doc_id']}::chunk_{chunk['chunk_id']}"
        raw_text = chunk["text"]

        meta = {
            "doc_id":             chunk.get("doc_id", ""),
            "chunk_id":           str(chunk.get("chunk_id", "")),
            "document_name":      chunk.get("document_name", ""),
            "page_numbers":       str(chunk.get("page_numbers", [])),
            "primary_topic":      chunk.get("primary_topic", "Research"),
            "crop_entities":      ", ".join(chunk.get("crop_entities", [])),
            "practice_entities":  ", ".join(chunk.get("practice_entities", [])),
            "region_entities":    ", ".join(chunk.get("region_entities", [])),
            "numeric_signals":    ", ".join(chunk.get("numeric_signals", [])),
            "relevance_score":    float(chunk.get("relevance_score", 0.0)),
            "char_count":         len(raw_text),
            "raw_text":           raw_text,  # preserve original for LLM prompts
        }

        # Build enriched document for embedding
        enriched = build_enriched_document(raw_text, meta)

        ids.append(uid)
        documents.append(enriched)
        metadatas.append(meta)

    # Load GPU model for ingestion
    try:
        gpu_embed_fn = NomicEmbeddingFunction(device="cuda", is_query=False)
    except Exception as e:
        logger.error(f"Failed to load GPU model: {e}. Falling back to CPU.")
        gpu_embed_fn = NomicEmbeddingFunction(device="cpu", is_query=False)

    # Smaller batches = gentler on CPU/GPU during embedding
    BATCH = 200
    total = 0
    for i in range(0, len(ids), BATCH):
        batch_docs = documents[i:i+BATCH]
        batch_embeddings = gpu_embed_fn(batch_docs)
        
        col.upsert(
            ids=ids[i:i+BATCH],
            documents=batch_docs,
            metadatas=metadatas[i:i+BATCH],
            embeddings=batch_embeddings,
        )
        total += len(ids[i:i+BATCH])
        # Brief pause between batches to cool hardware
        if i + BATCH < len(ids):
            import time
            time.sleep(0.5)

    logger.info(f"Upserted {total} chunks (enriched embeddings) → collection now has {col.count()} docs")
    return total


def search(query: str,
           top_k: int = 5,
           topic_filter: Optional[str] = None,
           min_relevance: float = 0.0) -> List[Dict[str, Any]]:
    """
    Semantic search with optional metadata filters.

    Returns list of dicts:
        { text, score, doc_id, chunk_id, primary_topic, ... }
    """
    col = get_collection()

    where_filter = None
    conditions = []
    if topic_filter:
        conditions.append({"primary_topic": {"$eq": topic_filter}})
    if min_relevance > 0:
        conditions.append({"relevance_score": {"$gte": min_relevance}})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    try:
        results = col.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.warning(f"ChromaDB query failed: {e}. Resetting client handle and retrying...")
        global _collection, _client
        _collection = None
        _client = None
        try:
            col = get_collection()
            results = col.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as retry_err:
            logger.error(f"ChromaDB query retry failed: {retry_err}")
            return []

    hits = []
    if results and results["ids"] and results["ids"][0]:
        for idx, doc_id in enumerate(results["ids"][0]):
            dist = results["distances"][0][idx] if results["distances"] else 1.0
            score = 1.0 - dist          # cosine distance → similarity

            meta = results["metadatas"][0][idx] if results["metadatas"] else {}
            text = results["documents"][0][idx] if results["documents"] else ""

            hits.append({
                "text": text,
                "score": round(score, 4),
                **meta,
            })

    # Fallback: if strict topic filter produced 0 hits, search without topic filter
    if not hits and where_filter:
        logger.info(f"Strict topic filter produced 0 hits. Falling back to unfiltered search for: '{query[:50]}'")
        try:
            fallback_results = col.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            if fallback_results and fallback_results["ids"] and fallback_results["ids"][0]:
                for idx, doc_id in enumerate(fallback_results["ids"][0]):
                    dist = fallback_results["distances"][0][idx] if fallback_results["distances"] else 1.0
                    score = 1.0 - dist
                    meta = fallback_results["metadatas"][0][idx] if fallback_results["metadatas"] else {}
                    text = fallback_results["documents"][0][idx] if fallback_results["documents"] else ""
                    hits.append({
                        "text": text,
                        "score": round(score, 4),
                        **meta,
                    })
        except Exception as fb_err:
            logger.warning(f"Fallback search failed: {fb_err}")

    return hits


def collection_stats() -> Dict[str, Any]:
    """Quick health-check: collection size, distinct docs, topics."""
    col = get_collection()
    count = col.count()

    # Sample a few records to get topic distribution
    sample = col.get(limit=min(count, 200), include=["metadatas"])
    topics = {}
    doc_names = set()
    for meta in (sample.get("metadatas") or []):
        t = meta.get("primary_topic", "Unknown")
        topics[t] = topics.get(t, 0) + 1
        doc_names.add(meta.get("document_name", ""))

    return {
        "total_chunks": count,
        "distinct_documents": len(doc_names),
        "topic_distribution": topics,
    }
