"""
hybrid_search.py
────────────────────────────────────────────────────────────
Production Hybrid Search Engine for KrishiPlan RAG.

Combines ChromaDB's Semantic Search with in-memory BM25 lexical search,
fused together using Reciprocal Rank Fusion (RRF).

The BM25 index is built once upon initialization (Singleton pattern)
to avoid rebuilding it on every API request.
────────────────────────────────────────────────────────────
"""

import math
import re
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter

from .vector_store import get_collection

logger = logging.getLogger(__name__)

# ── 1. Basic NLP Utilities for BM25 ──

STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

def tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, split by whitespace, and remove stop words."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


# ── 2. Lightweight BM25 Implementation ──

class BM25:
    def __init__(self, corpus: List[List[str]], k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size else 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        
        # Calculate DF and IDF
        df = Counter()
        for doc in corpus:
            self.doc_len.append(len(doc))
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                df[word] += 1
                
        for word, freq in df.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for i in range(self.corpus_size):
            doc_len = self.doc_len[i]
            if doc_len == 0:
                continue
            for word in query:
                if word not in self.doc_freqs[i]:
                    continue
                tf = self.doc_freqs[i][word]
                numerator = self.idf[word] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[i] += (numerator / denominator)
        return scores


# ── 3. Hybrid Search Engine ──

class HybridSearchEngine:
    def __init__(self):
        logger.info("[System] Initializing Hybrid Search Engine (Building BM25 Index)...")
        self.col = get_collection()
        
        # Fetch all chunks (manageable size in memory)
        all_docs = self.col.get(include=["metadatas"])
        
        self.doc_ids = all_docs["ids"]
        self.metadatas = all_docs["metadatas"]
        
        # Extract raw text from metadata for lexical indexing
        self.corpus_texts = []
        for meta in self.metadatas:
            text = meta.get("raw_text", "")
            self.corpus_texts.append(text)
            
        logger.info(f"[System] Tokenizing {len(self.corpus_texts)} chunks...")
        self.tokenized_corpus = [tokenize(text) for text in self.corpus_texts]
        self.bm25 = BM25(self.tokenized_corpus)
        logger.info("[System] BM25 Index Ready.")

    def search_bm25(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair IDs with scores and sort
        scored_docs = list(zip(self.doc_ids, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K non-zero chunks
        return [doc for doc in scored_docs[:top_k] if doc[1] > 0]

    def search_semantic(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        # Pure semantic search (NO topic filter, NO relevance_score filter)
        results = self.col.query(
            query_texts=[query],
            n_results=top_k,
            include=["distances"]
        )
        
        hits = []
        if results and results["ids"] and results["ids"][0]:
            for idx, doc_id in enumerate(results["ids"][0]):
                dist = results["distances"][0][idx] if results["distances"] else 1.0
                score = 1.0 - dist  # Convert distance to similarity
                hits.append((doc_id, score))
        return hits

    def hybrid_search(self, query: str, final_k: int = 6) -> List[Dict[str, Any]]:
        logger.info(f"[Hybrid] Executing search for: '{query[:50]}'")
        
        # 1. Get Top N from both methods (We fetch final_k * 3 for deeper fusion)
        fetch_k = final_k * 3
        bm25_hits = self.search_bm25(query, top_k=fetch_k)
        semantic_hits = self.search_semantic(query, top_k=fetch_k)
        
        # 2. Reciprocal Rank Fusion (RRF)
        # Give semantic hits a weight of 2.0 and BM25 a weight of 1.0
        K = 60
        fused_scores = {}
        
        for rank, (doc_id, _) in enumerate(bm25_hits, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (K + rank))
            
        for rank, (doc_id, _) in enumerate(semantic_hits, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (2.0 / (K + rank))
            
        # Sort by fused score
        ranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = [doc_id for doc_id, _ in ranked_results[:final_k]]
        
        # Fetch the actual metadata for the final Top K
        final_chunks = []
        if top_ids:
            final_data = self.col.get(ids=top_ids, include=["metadatas"])
            # Reorder them to match top_ids sequence
            id_to_meta = {uid: meta for uid, meta in zip(final_data["ids"], final_data["metadatas"])}
            for uid, score in ranked_results[:final_k]:
                if uid in id_to_meta:
                    meta = id_to_meta[uid]
                    # We store the final RRF score in "score" so the rest of the RAG pipeline sees it
                    meta["score"] = round(score, 4)
                    final_chunks.append(meta)
                    
        return final_chunks


# ── 4. Singleton Instance ──
_engine_instance = None

def get_hybrid_engine() -> HybridSearchEngine:
    """Returns the singleton instance of the HybridSearchEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = HybridSearchEngine()
    return _engine_instance
