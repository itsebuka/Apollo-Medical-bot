"""
Apollo Medical Triage System — FastAPI Application (Phase 2)
=============================================================
Author: Built for ADTC 2026 — Team: Eleogu Chukwuebuka Joseph

This is the heart of Apollo's backend. It:
1. Loads the Llama 3 GGUF model and embedding model ONCE at startup
2. Exposes a /chat endpoint that runs the full RAG pipeline
3. Streams tokens back to the client in real-time via Server-Sent Events (SSE)
4. Exposes a /health endpoint for diagnostics

Architecture: Single-process, single-worker. Do NOT run with multiple uvicorn
workers — loading the 4.5GB model in multiple processes would exceed RAM limits.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

import json
import re
import time
import asyncio
import logging
import functools
import copy
from contextlib import asynccontextmanager
import contextlib
from typing import AsyncGenerator

try:
    from app import config  # type: ignore
except ImportError:
    from . import config  # type: ignore

try:
    from app.pipeline import (  # type: ignore
        preprocess_query, validate_and_repair, build_system_context_block,
        ApolloResponse, Escalation,
    )
except ImportError:
    from .pipeline import (
        preprocess_query, validate_and_repair, build_system_context_block,
        ApolloResponse, Escalation,
    )

import sqlite3
import chromadb
import io
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # Will raise a clear error at upload time if missing

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# Structured logging is critical for debugging a black-box offline system.
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("apollo")

@contextlib.contextmanager
def time_profiler(label: str):
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"[PROFILE] {label} took {elapsed:.4f}s")


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# Pydantic models provide automatic validation and OpenAPI documentation.
# If the client sends a malformed request, FastAPI returns a 422 error
# automatically — no manual validation code needed.
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat request payload."""
    messages: list[dict] = Field(
        ...,
        description="The chat history. The last message must be the current user query."
    )
    # Optional: allow the client to request a specific number of context chunks
    n_results: int = Field(
        default=5,
        ge=1,
        le=7,
        description="Number of RAG context chunks to retrieve (1-7)"
    )
    uploaded_context: str | None = Field(
        default=None,
        description="Optional session-scoped document text uploaded by the user"
    )
    domain_filter: str | None = Field(
        default=None,
        description=(
            "Restrict ChromaDB retrieval to a specific knowledge domain (e.g. 'virology'). "
            "Maps to the 'domain' metadata field set during ingestion. "
            "None or 'all' means search the entire corpus."
        )
    )
    use_hyde: bool = Field(
        default=False,
        description=(
            "If True, run HyDE (Hypothetical Document Embedding) expansion when baseline "
            "retrieval quality is low. Enabled for Deep Research mode; disabled for Triage mode "
            "to save the 10-15s HyDE inference cost."
        )
    )

class TitleRequest(BaseModel):
    query: str = Field(..., max_length=1000)

class TitleResponse(BaseModel):
    title: str

class HealthResponse(BaseModel):
    """Health check response payload."""
    status: str
    model_loaded: bool
    embedding_model_loaded: bool
    vector_db_document_count: int
    model_path: str
    uptime_seconds: float


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION LIFESPAN (STARTUP / SHUTDOWN)
# The @asynccontextmanager lifespan pattern is the modern FastAPI way to
# handle resource initialization and cleanup. It replaces the deprecated
# @app.on_event("startup") decorator.
#
# WHY LOAD AT STARTUP?
# Loading the 4.5GB GGUF model takes 8-12 seconds. If we loaded on the first
# request, the user would experience a 12-second hang on their very first
# query. By loading at server startup, the first request is fast.
# ─────────────────────────────────────────────────────────────────────────────

_startup_time = time.time()  # Track server uptime

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Code before 'yield' runs at startup.
    Code after 'yield' runs at shutdown.
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  APOLLO — Medical Triage System Starting Up")
    logger.info("=" * 60)

    # Validate model file exists before attempting to load
    if not config.MODEL_PATH.exists():
        logger.error(f"GGUF model not found at: {config.MODEL_PATH}")
        logger.error("Run 'bash download_model.sh' from the repo root first.")
        raise FileNotFoundError(f"Model file missing: {config.MODEL_PATH}")

    # Validate ChromaDB exists (ingest.py must have been run)
    if not config.CHROMA_DB_DIR.exists():
        logger.error(f"ChromaDB not found at: {config.CHROMA_DB_DIR}")
        logger.error("Run 'python ingest.py' from the backend directory first.")
        raise FileNotFoundError(f"Vector DB missing: {config.CHROMA_DB_DIR}")

    # ── Load Llama 3 GGUF Model ───────────────────────────────────────────────
    logger.info(f"Loading LLM: {config.MODEL_PATH.name}")
    logger.info(f"  Config: n_ctx={config.LLM_CONFIG['n_ctx']}, "
                f"n_threads={config.LLM_CONFIG['n_threads']}, "
                f"n_gpu_layers={config.LLM_CONFIG['n_gpu_layers']}")

    t = time.time()
    try:
        config.llm_instance = Llama(
            model_path=str(config.MODEL_PATH),
            **config.LLM_CONFIG,
        )
        logger.info(f"LLM loaded in {time.time() - t:.1f}s ✓")
    except Exception as e:
        logger.warning(f"Initial LLM load failed ({e}). Retrying with flash_attn=False & use_mmap=True...")
        try:
            fallback_config = {**config.LLM_CONFIG, "flash_attn": False, "use_mmap": True}
            config.llm_instance = Llama(
                model_path=str(config.MODEL_PATH),
                **fallback_config,
            )
            logger.info(f"LLM loaded (fallback config) in {time.time() - t:.1f}s ✓")
        except Exception as e2:
            logger.warning(f"Second LLM load attempt failed ({e2}). Retrying with minimal default parameters...")
            config.llm_instance = Llama(
                model_path=str(config.MODEL_PATH),
                n_ctx=config.LLM_CONFIG.get("n_ctx", 4096),
                n_threads=config.LLM_CONFIG.get("n_threads", 4),
                use_mmap=True,
                verbose=False,
            )
            logger.info(f"LLM loaded (minimal fallback) in {time.time() - t:.1f}s ✓")

    # ── Load Embedding Model ──────────────────────────────────────────────────
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    t = time.time()
    config.embedding_model_instance = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    logger.info(f"Embedding model loaded in {time.time() - t:.1f}s ✓")

    # ── Load Cross-Encoder Model ──────────────────────────────────────────────
    logger.info(f"Loading cross-encoder model: {config.CROSS_ENCODER_MODEL_NAME}")
    t = time.time()
    config.cross_encoder_instance = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME)
    logger.info(f"Cross-encoder loaded in {time.time() - t:.1f}s ✓")

    # ── Connect to ChromaDB ───────────────────────────────────────────────────
    logger.info(f"Connecting to ChromaDB at: {config.CHROMA_DB_DIR}")
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
    config.chroma_collection_instance = chroma_client.get_collection(
        name=config.CHROMA_COLLECTION_NAME
    )
    doc_count = config.chroma_collection_instance.count()
    logger.info(f"ChromaDB ready. Collection '{config.CHROMA_COLLECTION_NAME}' "
                f"has {doc_count} documents ✓")

    logger.info("=" * 60)
    logger.info("  Apollo is READY. Listening on http://0.0.0.0:8000")
    logger.info("=" * 60)

    # Hand control to FastAPI — server handles requests from here
    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("Apollo shutting down. Releasing model memory...")
    # llama_cpp automatically frees the model when the process exits,
    # but explicit deletion is good practice
    del config.llm_instance
    del config.embedding_model_instance
    del config.cross_encoder_instance
    logger.info("Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Apollo Medical Triage API",
    description=(
        "Offline RAG-powered medical triage for NCDs in Nigeria. "
        "Powered by Llama 3 8B Q4_K_M + ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",   # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc", # ReDoc UI at http://localhost:8000/redoc
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
# Cross-Origin Resource Sharing: allows the React frontend (running on port 5173
# or 3000) to make requests to this API (running on port 8000).
# Without this, the browser blocks all cross-origin requests.
# In production, restrict allow_origins to your exact frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# RAG PIPELINE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Simple Clinical Query Expander
CLINICAL_EXPANSION_DICT = {
    "hiv": "human immunodeficiency virus",
    "tb": "tuberculosis",
    "amr": "antimicrobial resistance",
    "pk": "pharmacokinetics",
    "pd": "pharmacodynamics",
    "moa": "mechanism of action",
    "hx": "history",
    "tx": "treatment",
    "rx": "prescription",
    "dx": "diagnosis",
}

def expand_query(query: str) -> str:
    """Normalizes and expands clinical abbreviations prior to embedding, and appends high-yield terms for biochemical mechanism queries."""
    words = query.split()
    expanded = []
    for w in words:
        clean_w = w.lower().strip(".,!?")
        if clean_w in CLINICAL_EXPANSION_DICT:
            expanded.append(CLINICAL_EXPANSION_DICT[clean_w])
        else:
            expanded.append(w)
            
    base_expanded = " ".join(expanded)
    q_lower = query.lower()
    
    # Mechanism Query Expansion Target: append high-yield enzymatic terms to guide vector & sparse retrieval
    if any(k in q_lower for k in ["biochemical mechanism", "enzymatic mechanism", "pathogenicity", "mode of action", "enhancin", "cleavage"]):
        base_expanded += " metalloproteinase zinc endopeptidase mucin peritrophic membrane degradation cleavage substrate catalytic"
        
    return base_expanded


def _clean_source_label(source: str) -> str:
    """
    Converts a raw source string like:
        'molecular-virology-moses-p-adoga-2347.pdf (Page 86)'
    into a clean, publication-grade document title label:
        'Molecular Virology (Moses P. Adoga) (Page 86)'
    """
    import re as _re

    # Split page suffix if present, e.g. "file.pdf (Page 86)" -> "file.pdf", "(Page 86)"
    page_suffix = ""
    page_match = _re.search(r"(\(Page[^)]*\))", source)
    if page_match:
        page_suffix = " " + page_match.group(1)
        source_base = source[:page_match.start()].strip()
    else:
        source_base = source.strip()

    # Strip file extension and numeric database suffixes
    source_base = _re.sub(r"\.(pdf|txt)$", "", source_base, flags=_re.IGNORECASE)
    source_base = _re.sub(r"[-_]\d{3,5}$", "", source_base)

    # Replace hyphens and underscores with spaces
    source_base = source_base.replace("-", " ").replace("_", " ")

    # Filter out bogus standalone digits or database artifacts
    tokens = [t for t in source_base.split() if not (_re.fullmatch(r"\d{4,}", t) and (int(t) < 1900 or int(t) > 2030))]
    cleaned = " ".join(tokens).strip()

    # Title-case for readability
    cleaned = cleaned.title() if cleaned else source_base.strip()

    return f"{cleaned}{page_suffix}"



def is_casual_query(query: str) -> bool:
    """Detects simple conversational greetings to bypass RAG retrieval entirely.

    Three-stage matching handles punctuation variations, non-English greetings,
    and informal spellings: normalise → exact-match → root-word check.
    """
    # Stage 0: Hard exit for anything too long to be a pure greeting
    raw_words = query.split()
    if len(raw_words) > 10:
        return False

    # Stage 1: Normalise — remove ALL punctuation, lowercase, collapse whitespace
    # "Hey, good afternoon!" → "hey good afternoon"
    # "Hello!!" → "hello"  |  "hi   there" → "hi there"
    q_clean = re.sub(r"[^\w\s]", " ", query.lower())
    q_clean = re.sub(r"\s+", " ", q_clean).strip()


    # Stage 2: Exact-phrase match against an expanded normalised phrase list
    exact_matches = {
        # English greetings
        "hi", "hey", "hello", "yo", "sup", "howdy", "hiya", "heya",
        "greetings", "salutations",
        # Time-based
        "good morning", "good afternoon", "good evening", "good day", "good night",
        # With Apollo
        "hi apollo", "hey apollo", "hello apollo",
        "good morning apollo", "good afternoon apollo", "good evening apollo",
        # With punctuation normalised
        "morning", "afternoon", "evening",
        # Informal / social & compounds
        "how are you", "how are you doing", "how is it going",
        "how do you do", "whats up", "whats good", "whats popping",
        "hows it going", "hows everything",
        "hi how are you", "hi how are you doing",
        "hello how are you", "hello how are you doing",
        "hey how are you", "hey how are you doing",
        # Pidgin / informal Nigerian
        "how far", "how body", "oga", "nna",
        # Extended greetings with identifiers
        "hey there", "hello there", "hi there",
        "hey good afternoon", "hey good morning", "hey good evening",
        "hi good afternoon", "hi good morning", "hi good evening",
        "hello good afternoon", "hello good morning", "hello good evening",
    }
    if q_clean in exact_matches:
        return True

    # Stage 3: Root-word check — only if ≤ 8 words and NO medical term present
    words = q_clean.split()
    if len(words) > 8:
        return False

    # If query contains any clinical/medical term, it is NOT casual regardless of greeting start
    # Keep this strictly to domain-specific clinical words (avoid generic stop/auxiliary words)
    medical_terms = {
        "treat", "treatment", "cure", "diagnosis", "diagnose", "symptom", "symptoms",
        "drug", "medication", "medicine", "dose", "dosage", "antibiotic",
        "pain", "fever", "sick", "ill", "disease", "infection", "virus",
        "bacteria", "fungal", "parasite", "cancer", "diabetes", "hypertension",
        "explain", "describe", "define", "list", "outline", "compare", "contrast",
        "mechanism", "pathway", "structure", "function", "effect", "patient", "clinical",
    }
    if any(w in medical_terms for w in words):
        return False

    # Check if the FIRST word is a known greeting root
    greeting_roots = {
        "hi", "hey", "hello", "good", "howdy", "greetings", "salut",
        "yo", "sup", "hiya", "heya", "morning", "afternoon", "evening",
    }
    if words and words[0] in greeting_roots:
        return True

    return False


def decompose_comparative_query(query: str) -> list[str]:
    """
    Detects comparison/contrast queries and decomposes them into entity-level sub-queries.

    A single embedding vector cannot simultaneously represent two distinct medical
    entities (e.g., M1 protein and CM2 protein). This function detects comparative
    markers and generates a targeted sub-query per entity so that the retriever
    fetches high-quality context for BOTH sides before the cross-encoder merges them.

    E.g., "Structural difference between M1 and CM2 in Influenza C" ->
          ["M1 protein Influenza C", "CM2 protein Influenza C", <original>]
    """
    q_lower = query.lower()
    comparative_markers = [' vs ', ' vs. ', ' versus ', ' compared to ', 'difference between ', 'distinguish between ']

    if not any(m in q_lower for m in comparative_markers):
        return [query]  # Not a comparison — single retrieval is fine

    sub_queries = [query]  # Original always included as an anchor

    for marker in [' vs ', ' vs. ', ' versus ', ' compared to ']:
        if marker in q_lower:
            idx = q_lower.find(marker)
            left_text = query[:idx].strip()
            right_text = query[idx + len(marker):].strip()

            # Take the last 5 words of the left entity, first 5 of the right entity
            left_entity = ' '.join(left_text.split()[-5:])
            right_entity = ' '.join(right_text.split()[:5])

            # Append trailing context words from the full query for grounding
            tail_context = ' '.join(query.split()[-4:]) if len(query.split()) > 8 else ''

            if left_entity:
                sub_queries.append(f"{left_entity} {tail_context}".strip())
            if right_entity:
                sub_queries.append(f"{right_entity} {tail_context}".strip())
            break  # Only process the first comparative marker found

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in sub_queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


async def generate_hyde(query: str) -> str:
    """Generates a brief hypothetical clinical reference paragraph to improve dense vector match."""
    if not config.llm_instance:
        return query
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a clinical expert. Write a brief hypothetical 2-sentence clinical reference paragraph "
        f"answering the user's query using exact medical terms.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: config.llm_instance(
                prompt,
                max_tokens=60,
                temperature=0.3,
                stop=["<|eot_id|>"]
            )
        )
        synth_text = res["choices"][0]["text"].strip()
        if synth_text:
            logger.info(f"[HyDE] Generated synthetic document: {synth_text[:70]}...")
            return synth_text
    except Exception as e:
        logger.error(f"HyDE generation error: {e}")
    return query


def _predict_cross_encoder(query: str, candidates: list[dict]) -> list[dict]:
    """Re-ranks candidates using the cross-encoder. Returns NEW dicts to avoid
    mutating the LRU-cached originals (cache corruption guard)."""
    if not candidates or not config.cross_encoder_instance:
        return candidates
    try:
        # Score up to 1300 chars of each parent chunk.
        # Parent chunks are ~1500 chars; the old 900-char limit cut off the final third
        # where key mechanistic sentences (e.g. "host DNA repair corrects RT errors") often appear.
        pairs = [[query, ctx["text"][:1300]] for ctx in candidates]
        scores = config.cross_encoder_instance.predict(pairs)
        # Build NEW dicts — never mutate the cached originals
        scored = []
        for i, ctx in enumerate(candidates):
            new_ctx = copy.copy(ctx)  # shallow copy is enough (all values are immutable primitives)
            new_ctx["cross_encoder_score"] = float(scores[i])
            scored.append(new_ctx)
        return sorted(scored, key=lambda x: x["cross_encoder_score"], reverse=True)
    except Exception as e:
        logger.error(f"[CrossEncoder] Re-ranking failed: {e} — returning unranked candidates")
        return candidates


@functools.lru_cache(maxsize=100)
def _sync_candidate_retrieval(query: str, n_results: int, search_query_override: str | None = None, domain_filter: str | None = None) -> tuple:
    """Hybrid dense + sparse retrieval with RRF fusion. LRU-cached by (query, n_results, override, domain_filter).

    Returns a TUPLE (not a list) so the lru_cache holds an immutable reference.
    Callers must wrap with list() before mutating or slicing.
    """
    # Guard against uninitialised models (e.g. called before startup completes)
    if not config.embedding_model_instance or not config.chroma_collection_instance:
        logger.error("[Retrieval] Models not initialised — returning empty candidates")
        return ()

    search_text = search_query_override if search_query_override else query
    expanded_query = expand_query(search_text)

    # Initial retrieval pool expansion: initial_k = 12 candidates prior to Cross-Encoder re-ranking
    total_docs = config.chroma_collection_instance.count()
    search_k = min(max(12, n_results * 4), max(1, total_docs))

    # Build optional ChromaDB where filter for domain scoping
    # When domain_filter is set (e.g. 'virology'), only chunks from that domain are retrieved.
    # This makes the scope dropdown in the UI functionally meaningful.
    chroma_where: dict | None = None
    if domain_filter and domain_filter.lower() not in ('all', ''):
        chroma_where = {"domain": {"$eq": domain_filter.lower()}}
        logger.info(f"[Retrieval] Domain filter active: domain='{domain_filter}'")

    # ── 1. DENSE SEARCH (ChromaDB) ──────────────────────────────────────────
    try:
        query_embedding = config.embedding_model_instance.encode(
            [expanded_query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        chroma_query_kwargs: dict = dict(
            query_embeddings=query_embedding,
            n_results=search_k,
            include=["metadatas", "distances"],
        )
        if chroma_where:
            chroma_query_kwargs["where"] = chroma_where

        chroma_results = config.chroma_collection_instance.query(**chroma_query_kwargs)
    except Exception as e:
        logger.error(f"[ChromaDB] Dense search failed: {e}")
        chroma_results = {"metadatas": [[]], "distances": [[]]}

    DISTANCE_THRESHOLD = 0.80
    chroma_parents = []
    seen_chroma_parents = set()

    for meta, dist in zip(chroma_results["metadatas"][0], chroma_results["distances"][0]):
        if dist > DISTANCE_THRESHOLD:
            continue
        parent_id = meta.get("parent_id")
        if not parent_id or parent_id in seen_chroma_parents:
            continue
        seen_chroma_parents.add(parent_id)
        chroma_parents.append({
            "text": meta.get("parent_text", ""),
            "source": f"{meta.get('source_file', 'unknown')} (Page {meta.get('page_number', '?')})",
            "similarity": round(1 - dist, 4),
        })

    # ── 2. SPARSE SEARCH (SQLite FTS5) ──────────────────────────────────────
    # Use a context manager so the connection is ALWAYS closed, even on exception
    sqlite_parents = []
    try:
        # Strip quotes/apostrophes and keep only alphanumeric tokens
        raw_tokens = [w for w in search_text.replace('"', '').replace("'", "").split() if w.isalnum()]
        if raw_tokens:
            # Build an AND-conjunction query: all tokens must be present in the chunk.
            # This prevents "hepatitis b virus" from matching chunks that only contain "b"
            # (FTS5 bare word list is OR by default — AND is correct for clinical precision).
            fts_and_query = " AND ".join(raw_tokens)

            with sqlite3.connect(config.REPO_ROOT / "backend" / "fts.db") as conn:
                c = conn.cursor()
                # Check if domain column exists for backwards-compatible domain filtering
                use_domain = False
                if domain_filter and domain_filter.lower() not in ('all', ''):
                    try:
                        c.execute("PRAGMA table_info(chunks)")
                        cols = [r[1] for r in c.fetchall()]
                        if "domain" in cols:
                            use_domain = True
                    except Exception:
                        use_domain = False

                # First attempt: strict AND — all terms must appear
                if use_domain:
                    c.execute('''
                        SELECT parent_text, source_file, page_number
                        FROM chunks
                        WHERE chunks MATCH ? AND domain = ?
                        ORDER BY rank
                        LIMIT ?
                    ''', (fts_and_query, domain_filter.lower(), search_k))
                else:
                    c.execute('''
                        SELECT parent_text, source_file, page_number
                        FROM chunks
                        WHERE chunks MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    ''', (fts_and_query, search_k))
                rows = c.fetchall()

                # Fallback: if AND returned nothing (e.g. very rare multi-word combo),
                # retry with the original OR-style bare query to avoid empty sparse results
                if not rows and len(raw_tokens) > 1:
                    fts_or_query = " ".join(raw_tokens)
                    logger.info(f"[FTS5] AND query returned 0 results — retrying with OR fallback")
                    if use_domain:
                        c.execute('''
                            SELECT parent_text, source_file, page_number
                            FROM chunks
                            WHERE chunks MATCH ? AND domain = ?
                            ORDER BY rank
                            LIMIT ?
                        ''', (fts_or_query, domain_filter.lower(), search_k))
                    else:
                        c.execute('''
                            SELECT parent_text, source_file, page_number
                            FROM chunks
                            WHERE chunks MATCH ?
                            ORDER BY rank
                            LIMIT ?
                        ''', (fts_or_query, search_k))
                    rows = c.fetchall()

                seen_sqlite_parents = set()
                for row in rows:
                    p_text = row[0]
                    if p_text in seen_sqlite_parents:
                        continue
                    seen_sqlite_parents.add(p_text)
                    sqlite_parents.append({
                        "text": p_text,
                        "source": f"{row[1]} (Page {row[2]})",
                        # Use 0.75 (not 0.99) so FTS hits don't always dominate the RRF merge
                        "similarity": 0.75,
                    })
    except Exception as e:
        logger.error(f"[FTS5] Sparse search failed: {e}")

    # ── 3. RECIPROCAL RANK FUSION (RRF) ──────────────────────────────────────
    rrf_scores: dict[str, float] = {}
    merged_contexts: dict[str, dict] = {}
    k = 60

    for rank, ctx in enumerate(chroma_parents):
        key = ctx["text"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        merged_contexts[key] = ctx

    for rank, ctx in enumerate(sqlite_parents):
        key = ctx["text"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in merged_contexts:
            merged_contexts[key] = ctx

    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    # Return a TUPLE so the lru_cache holds an immutable reference — never a mutable list
    return tuple(merged_contexts[key] for key in sorted_keys[:search_k])


def _sanitize_clinical_query(query: str) -> str:
    """
    Strips noise preambles like 'second attemp at asking the question:',
    'Apollo Triage Summary', previous output paste-backs, or conversational intros
    to isolate the core clinical question for high-precision RAG vector retrieval.
    """
    clean_q = query.strip()
    
    # If the user pasted back an old Apollo output, extract the actual question or header line
    if "Apollo Triage Summary" in clean_q or "Clinical Context Analysis" in clean_q:
        lines = [l.strip() for l in clean_q.splitlines() if l.strip()]
        for l in lines:
            if l.lower().startswith("question:") or l.lower().startswith("query:") or "biochemical mechanism" in l.lower():
                clean_q = re.sub(r"^(question|query):\s*", "", l, flags=re.IGNORECASE).strip()
                break
        else:
            for l in lines:
                if "apollo triage summary" not in l.lower() and "generated:" not in l.lower():
                    clean_q = l
                    break

    # Strip common attempt prefixes
    clean_q = re.sub(r"^(second|third|another)?\s*(attemp|attempt|try)\s*(at|of)?\s*(asking|running)?\s*(the)?\s*(question|query)?:?\s*", "", clean_q, flags=re.IGNORECASE)

    return clean_q.strip() or query


async def retrieve_context(query: str, n_results: int, search_query_override: str | None = None, domain_filter: str | None = None) -> list[dict]:
    """
    Advanced Hybrid Retrieval with Multi-Query Decomposition and Async Cross-Encoder Re-Ranking.

    For comparison/contrast queries, decomposes into entity-level sub-queries, retrieves
    candidates for each, merges and deduplicates the pool, then runs the cross-encoder
    against the *original* query to select the truly best chunks.

    For all other queries, falls back to single-query retrieval (no overhead).
    All sub-operations are individually wrapped in try/except so a partial failure
    never kills the whole request.
    """
    try:
        query = _sanitize_clinical_query(query)
        if search_query_override:
            # HyDE path — use the provided override directly, no decomposition
            # list() unpacks the cached tuple into a fresh mutable list for downstream use
            candidates = list(_sync_candidate_retrieval(query, n_results, search_query_override, domain_filter))
        else:
            try:
                sub_queries = decompose_comparative_query(query)
            except Exception as e:
                logger.error(f"[MultiQuery] decompose_comparative_query failed: {e} — using original query")
                sub_queries = [query]

            if len(sub_queries) == 1:
                # Fast path: single direct retrieval (no comparative keywords detected)
                candidates = list(_sync_candidate_retrieval(query, n_results, None, domain_filter))
            else:
                # Multi-query path: retrieve per sub-query, merge, deduplicate
                logger.info(f"[MultiQuery] Decomposed into {len(sub_queries)} sub-queries for comparative retrieval")
                merged: dict[str, dict] = {}
                for sub_q in sub_queries:
                    try:
                        sub_candidates = list(_sync_candidate_retrieval(sub_q, n_results, None, domain_filter))
                        for c in sub_candidates:
                            key = c['text'][:120]
                            if key not in merged:
                                merged[key] = c
                    except Exception as e:
                        logger.error(f"[MultiQuery] Sub-query '{sub_q[:40]}' failed: {e} — skipping")
                        continue
                candidates = list(merged.values())
                logger.info(f"[MultiQuery] Merged pool: {len(candidates)} unique candidates before reranking")

        if not candidates:
            return []

        loop = asyncio.get_running_loop()
        with time_profiler("Async Cross-Encoder Re-Ranking"):
            # Always rerank against the ORIGINAL query for faithfulness
            re_ranked = await loop.run_in_executor(None, _predict_cross_encoder, query, candidates)

        return re_ranked[:n_results]

    except Exception as e:
        logger.error(f"[retrieve_context] Unhandled error: {e} — returning empty context")
        return []


def build_prompt(
    messages: list[dict],
    context_chunks: list[dict],
    uploaded_context: str | None = None,
    system_context_block: str | None = None,
) -> list[dict]:
    """
    Construct the Llama 3 Instruct-formatted prompt with RAG context, optional
    uploaded user document, and injected clinical system context block.

    The system_context_block (from pipeline.build_system_context_block) contains
    the pre-resolved age-band, emergency flag, and substance protocol so the LLM
    is given these as explicit variables rather than recalling from training.
    """
    if not messages:
        # Should never happen via the validated endpoint, but guard defensively
        return [{"role": "system", "content": config.CASUAL_SYSTEM_PROMPT},
                {"role": "user", "content": "Hello"}]

    latest_query = messages[-1]["content"]
    top_sim = max([c.get("similarity", 0.0) for c in context_chunks]) if context_chunks else 0.0

    # ── LAYER 3 — CHITCHAT FALLBACK ROUTING ─────────────────────────────────
    # Threshold raised from 0.35 → 0.45: any retrieval scoring below this is
    # too speculative to pass to the medical system prompt.
    # context_chunks is already empty when Layer 1 or Layer 2 fired above.
    if top_sim < 0.45 and not uploaded_context:
        final_messages = [{"role": "system", "content": config.CASUAL_SYSTEM_PROMPT}]
        history = messages[:-1][-4:]
        final_messages.extend(history)
        final_messages.append({"role": "user", "content": latest_query})
        return final_messages

    # ── STRICT MEDICAL ROUTING ──
    user_parts = []

    # Inject the pipeline's system context block first (if provided)
    if system_context_block:
        user_parts.append(system_context_block)

    if uploaded_context:
        user_parts.append(f"PATIENT/SESSION UPLOADED DOCUMENT:\n\n{uploaded_context}\n\n---")
        
    if context_chunks:
        context_block = "\n\n---\n\n".join([
            f"[Source File: {_clean_source_label(chunk['source'])} | Relevance: {chunk['similarity']:.2%}]\n{chunk['text']}"
            for chunk in context_chunks
        ])
        user_parts.append(f"CLINICAL CONTEXT (retrieved from Apollo's knowledge base):\n\n{context_block}\n\n---")
    elif not uploaded_context:
        user_parts.append(
            f"SYSTEM DIRECTIVE: Our local RAG database did not return any localized clinical context "
            f"that passed the relevance threshold for this query. "
            f"You must inform the user that Apollo lacks specific localized guidelines for this exact question, "
            f"but you may provide general, safe triage advice based on standard medical knowledge. "
            f"Ensure you emphasize the need to consult a physician."
        )

    user_parts.append(
        f"PATIENT/CLINICIAN QUERY: {latest_query}\n\n"
        f"Based strictly on the clinical context above and your medical knowledge, "
        f"provide a thorough, accurate, and actionable response using the strict 4-part schema."
    )

    user_content = "\n\n".join(user_parts)
    final_messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    
    # Add history (last 4 turns)
    history = messages[:-1][-4:]
    final_messages.extend(history)
    final_messages.append({"role": "user", "content": user_content})
    
    return final_messages


async def stream_llm_response(messages: list[dict], max_tokens: int = 2048) -> AsyncGenerator[str, None]:
    """
    Stream the LLM response token-by-token with dynamic max_tokens budget as Server-Sent Events (SSE).
    """
    if not config.llm_instance:
        error_payload = json.dumps({"type": "error", "content": "LLM not loaded — cannot generate response."})
        yield f"data: {error_payload}\n\n"
        return
    try:
        start_time = time.time()
        gen_kwargs = {**config.GENERATION_CONFIG, "max_tokens": max_tokens}
        stream = config.llm_instance.create_chat_completion(
            messages=messages,
            stream=True,
            **gen_kwargs,
        )

        loop = asyncio.get_running_loop()
        stream_iter = iter(stream)

        def _get_next_chunk():
            try:
                return next(stream_iter)
            except StopIteration:
                return None

        token_count = 0
        while True:
            chunk = await loop.run_in_executor(None, _get_next_chunk)
            if chunk is None:
                break

            # Each chunk from llama_cpp has this structure:
            # {"choices": [{"delta": {"content": "token_text"}, "finish_reason": null}]}
            delta = chunk["choices"][0]["delta"]
            finish_reason = chunk["choices"][0].get("finish_reason")

            # Extract the token text from the delta
            token_text = delta.get("content", "")

            if token_text:
                token_count += 1
                # Yield the token as a JSON-encoded SSE message
                payload = json.dumps({"type": "token", "content": token_text})
                yield f"data: {payload}\n\n"

            # When the model signals it's done, send a termination event
            if finish_reason is not None:
                elapsed = time.time() - start_time
                tps = token_count / elapsed if elapsed > 0 else 0

                end_payload = json.dumps({
                    "type": "end",
                    "finish_reason": finish_reason,
                    "tokens_generated": token_count,
                })
                yield f"data: {end_payload}\n\n"
                logger.info(f"Response complete: {token_count} tokens, reason={finish_reason}")
                logger.info(f"[PROFILE] Total LLM Generation Time: {elapsed:.4f}s ({tps:.1f} tokens/sec)")
                break

    except Exception as e:
        # Send error as SSE so the frontend can display it gracefully
        logger.error(f"LLM generation error: {e}", exc_info=True)
        error_payload = json.dumps({"type": "error", "content": str(e)})
        yield f"data: {error_payload}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    System health check endpoint.
    The frontend polls this on load to verify the backend is alive and the
    model is loaded before enabling the chat interface.
    """
    if config.llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM not yet loaded")

    doc_count = 0
    if config.chroma_collection_instance:
        doc_count = config.chroma_collection_instance.count()

    return HealthResponse(
        status="operational",
        model_loaded=config.llm_instance is not None,
        embedding_model_loaded=config.embedding_model_instance is not None,
        vector_db_document_count=doc_count,
        model_path=str(config.MODEL_PATH),
        uptime_seconds=round(time.time() - _startup_time, 1),
    )


@app.post("/generate_title", response_model=TitleResponse, tags=["Inference"])
async def generate_title(request: TitleRequest):
    """Generates a concise title from the user's initial query. Runs in executor to avoid blocking the event loop."""
    if config.llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM not loaded")

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a title generator. Summarize the user's query into a concise 2 to 4 word title. "
        f"Do not include quotes, punctuation, or conversational text. Just the title."
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{request.query}"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: config.llm_instance(
                prompt,
                max_tokens=8,
                temperature=0.3,
                stop=["<|eot_id|>"]
            )
        )
        title = res["choices"][0]["text"].strip(' "\'.\n')
        if not title:
            title = "New Chat"
        return TitleResponse(title=title)
    except Exception as e:
        logger.error(f"Title generation error: {e}")
        return TitleResponse(title="New Chat")


@app.post("/chat", tags=["Inference"])
async def chat(request: ChatRequest):
    """
    Primary RAG inference endpoint with HyDE expansion, Fallback retry, and Dynamic Token Allocation.
    """
    if config.llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM not loaded. Wait for server startup.")

    if not request.messages:
        raise HTTPException(status_code=422, detail="messages list cannot be empty.")

    latest_query = request.messages[-1]["content"].strip()
    if not latest_query:
        raise HTTPException(status_code=422, detail="The last message must have non-empty content.")

    # ── Layer A: Pre-Processing Pipeline (pipeline.py) ─────────────────────────
    # preprocess_query() does:
    #   1. Strip test/benchmark prefixes (regex, never seen by LLM)
    #   2. Deterministic age extraction (normalized to months)
    #   3. Red-flag pre-scan against clinical_protocol.yaml
    #   4. Resolve age-band slice from protocol config
    structured_query = preprocess_query(latest_query)
    if structured_query.cleaned_input != latest_query:
        # Patch messages list so cleaned query flows through RAG & prompt builder
        request.messages[-1] = {**request.messages[-1], "content": structured_query.cleaned_input}
        latest_query = structured_query.cleaned_input

    logger.info(f"[CHAT] Query received ({len(latest_query)} chars): {latest_query[:80]}...")

    # ── Step 0: Fast Chitchat Detection ──────────────────────────────────────
    t_rag = time.time()
    
    if is_casual_query(latest_query):
        logger.info("[CHAT] Casual greeting detected. Bypassing RAG.")
        context_chunks = []
        top_sim = 0.0
        rag_time = time.time() - t_rag
    else:
        # ── Step 1: Baseline RAG Retrieval ───────────────────────────────────────
        # Try a fast, direct hybrid search first
        domain_filter = request.domain_filter if request.domain_filter and request.domain_filter != 'all' else None
        if domain_filter:
            logger.info(f"[CHAT] Scope filter active: domain='{domain_filter}'")
        context_chunks = await retrieve_context(latest_query, request.n_results, domain_filter=domain_filter)
        top_sim = max([c.get("similarity", 0.0) for c in context_chunks]) if context_chunks else 0.0

        # ── Step 2: Lazy HyDE Expansion (Only if baseline is poor AND use_hyde=True) ─────
        # HyDE is gated on request.use_hyde (set by the frontend mode toggle):
        #   - Deep Research mode → use_hyde=True  → HyDE runs when retrieval is weak
        #   - Triage mode        → use_hyde=False → HyDE skipped (saves 10-15s per query)
        # Short queries also skip HyDE: they're either greetings or trivial lookups.
        if request.use_hyde and top_sim < 0.65 and len(latest_query.split()) >= 5:
            logger.info(f"[HyDE] Deep Research mode — running HyDE (initial similarity {top_sim:.4f} < 0.65)...")
            with time_profiler("HyDE Generation"):
                hyde_text = await generate_hyde(latest_query)
                if hyde_text and hyde_text != latest_query:
                    hyde_chunks = await retrieve_context(latest_query, request.n_results, search_query_override=hyde_text, domain_filter=domain_filter)
                    hyde_top_sim = max([c.get("similarity", 0.0) for c in hyde_chunks]) if hyde_chunks else 0.0
                    if hyde_top_sim > top_sim:
                        logger.info(f"[HyDE] Rescued query! Improved similarity from {top_sim:.4f} to {hyde_top_sim:.4f}")
                        context_chunks = hyde_chunks
                        top_sim = hyde_top_sim
        elif not request.use_hyde:
            logger.info("[HyDE] Skipped — Triage Mode (use_hyde=False). Use Deep Research for HyDE.")

        rag_time = time.time() - t_rag

        # ── LAYER 2 SAFETY NET: Short query + poor RAG match → force casual ──────
        # Even if is_casual_query missed a greeting (e.g. non-standard phrasing),
        # a legitimate greeting will NEVER achieve a high similarity score against
        # medical textbooks. If the query is very short AND the best retrieval score
        # is below the casual ceiling, discard the retrieved chunks entirely so the
        # casual system prompt is used in build_prompt.
        CASUAL_WORD_LIMIT   = 6    # queries of ≤6 words are candidates
        CASUAL_SIM_CEILING  = 0.50 # below this, a short query is treated as chitchat
        q_word_count = len(latest_query.split())
        if q_word_count <= CASUAL_WORD_LIMIT and top_sim < CASUAL_SIM_CEILING:
            logger.info(
                f"[CASUAL SAFETY NET] Short query ({q_word_count}w, sim={top_sim:.3f}) "
                f"below threshold — forcing casual routing"
            )
            context_chunks = []
            top_sim = 0.0
    logger.info(
        f"[RAG]  Retrieved {len(context_chunks)} chunks in {rag_time:.3f}s | "
        f"Top similarity: {context_chunks[0]['similarity'] if context_chunks else 'N/A'}"
    )

    # ── Step 2: Dynamic Token Allocation ────────────────────────────────────
    # Detect complex queries that require longer, structured responses.
    # These must NOT be penalized by the short-query 512-token budget, or
    # mechanism explanations will be truncated mid-answer.
    q_lower_budget = latest_query.lower()
    is_complex_query = any(w in q_lower_budget for w in [
        'compare', 'contrast', ' vs ', 'versus', 'difference between',
        'mechanism', 'explain', 'describe', 'how does', 'why is', 'why does',
        'pathway', 'synthesis', 'translational', 'structural', 'distinguish',
        'what are the', 'enumerate', 'list all', 'outline',
    ])

    q_len = len(latest_query)
    if is_complex_query:
        # Complex mechanism/comparison questions get 2560 max tokens so reference lists are never truncated
        dynamic_max_tokens = 2560
    elif q_len < 80 and top_sim > 0.75:
        dynamic_max_tokens = 1024
    elif q_len < 250 or top_sim > 0.50:
        dynamic_max_tokens = 1536
    else:
        dynamic_max_tokens = 2560
    logger.info(f"[BUDGET] Allocated dynamic max_tokens: {dynamic_max_tokens} (complex={is_complex_query})")

    # ── Step 3: Build the prompt with injected clinical context block ─────────
    # Inject the [SYSTEM CONTEXT] block from the structured query so the LLM
    # is given the correct age-band threshold, emergency flag, and substance
    # protocol as explicit variables — not recalled from training.
    system_context_block = build_system_context_block(structured_query)
    messages = build_prompt(
        request.messages, context_chunks,
        uploaded_context=request.uploaded_context,
        system_context_block=system_context_block,
    )

    # ── Step 4: Stream response with post-processing guardrails ──────────────
    # Wrap the raw SSE stream with Layer C (post-processing sanitizer + schema validator)
    # so every response is scrubbed of metadata, truncated at schema boundary,
    # and emergency-intercepted before reaching the frontend.
    async def stream_with_guardrails() -> AsyncGenerator[str, None]:
        """Collects the full LLM response, applies all post-processing guardrails,
        then re-emits it token-by-token to preserve the streaming UX."""
        full_response_parts: list[str] = []
        end_payload_cache: str | None = None

        async for sse_line in stream_llm_response(messages, max_tokens=dynamic_max_tokens):
            # Parse SSE line
            if not sse_line.startswith("data: "):
                continue
            try:
                event = json.loads(sse_line[6:])
            except json.JSONDecodeError:
                continue

            if event.get("type") == "token":
                full_response_parts.append(event["content"])
            elif event.get("type") == "end":
                end_payload_cache = sse_line
            elif event.get("type") == "error":
                # Pass errors straight through without buffering
                yield sse_line
                return

        # Apply the full deterministic pipeline to the fully assembled response
        raw_full = "".join(full_response_parts)
        if raw_full.strip():
            result = validate_and_repair(raw_full, structured_query)
            scrubbed = result.content  # Works for both ApolloResponse and Escalation
            if isinstance(result, Escalation):
                logger.error(
                    "[CHAT] Pipeline escalated: check=%s reason=%s",
                    result.failed_check, result.reason
                )
        else:
            scrubbed = raw_full

        # Re-emit as a stream of token events (preserves frontend streaming UX)
        # Chunk into ~8-char pieces to maintain smooth token-by-token display
        CHUNK_SIZE = 8
        for i in range(0, len(scrubbed), CHUNK_SIZE):
            chunk_text = scrubbed[i:i + CHUNK_SIZE]
            payload = json.dumps({"type": "token", "content": chunk_text})
            yield f"data: {payload}\n\n"

        # Re-emit the end event
        if end_payload_cache:
            yield end_payload_cache
        else:
            end_payload = json.dumps({"type": "end", "finish_reason": "stop", "tokens_generated": len(scrubbed.split())})
            yield f"data: {end_payload}\n\n"

    return StreamingResponse(
        stream_with_guardrails(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-RAG-Chunks-Retrieved": str(len(context_chunks)),
            "X-RAG-Top-Similarity": str(context_chunks[0]["similarity"] if context_chunks else 0),
        },
    )


@app.post("/upload_context", tags=["Context"])
async def upload_context(file: UploadFile = File(...)):
    """Extracts text from uploaded PDF or TXT files to use as session-scoped context."""
    filename = file.filename or "uploaded_doc"
    content_type = file.content_type or ""
    body = await file.read()

    extracted_text = ""
    try:
        if filename.lower().endswith(".pdf") or "pdf" in content_type:
            if PdfReader is None:
                raise HTTPException(status_code=500, detail="pypdf not installed — PDF upload unavailable.")
            reader = PdfReader(io.BytesIO(body))
            text_parts = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as page_err:
                    logger.warning(f"[UPLOAD] Skipped unreadable page in {filename}: {page_err}")
                    continue
            extracted_text = "\n\n".join(text_parts)
        else:
            extracted_text = body.decode("utf-8", errors="ignore")

        extracted_text = extracted_text.strip()
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from document.")

        # Truncate to ~3000 chars max to fit safely in context window
        was_truncated = len(extracted_text) > 3000
        if was_truncated:
            extracted_text = extracted_text[:3000] + "\n\n...[Uploaded Document Truncated]"

        # Clear the retrieval LRU cache so any prior cached results that predate
        # this upload don't persist. The new document should influence fresh retrievals.
        _sync_candidate_retrieval.cache_clear()
        logger.info(f"[UPLOAD] LRU cache cleared after document upload.")

        logger.info(f"[UPLOAD] Extracted {len(extracted_text)} chars from {filename} (truncated={was_truncated})")
        return {"filename": filename, "text": extracted_text, "truncated": was_truncated}
    except HTTPException:
        raise  # Re-raise FastAPI HTTP errors unchanged
    except Exception as e:
        logger.error(f"Error processing uploaded document {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.get("/", tags=["System"])
async def root():
    """API root — quick confirmation the server is running."""
    return {
        "system": "Apollo Medical Triage API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "challenge": "ADTC 2026 — healthcare_medical",
    }


@app.get("/domains", tags=["System"])
async def list_domains():
    """
    Returns a sorted list of distinct knowledge domains present in the ChromaDB collection.

    The frontend scope-selector uses this to dynamically populate its folder list.
    This means adding a new PDF to a new domain subfolder and re-running ingest.py
    will automatically appear as a new scope option in the UI — no code changes needed.
    """
    if config.chroma_collection_instance is None:
        raise HTTPException(status_code=503, detail="ChromaDB not initialised.")

    try:
        # ChromaDB does not have a native DISTINCT query, so we sample all metadata
        # (metadatas only — no documents or embeddings to keep this cheap) and
        # extract unique domain values in Python.
        all_meta = config.chroma_collection_instance.get(include=["metadatas"])
        domains: set[str] = set()
        for meta in all_meta.get("metadatas", []):
            domain = meta.get("domain", "").strip()
            if domain:
                domains.add(domain)

        sorted_domains = sorted(domains, key=str.lower)
        logger.info(f"[DOMAINS] Found {len(sorted_domains)} distinct domains: {sorted_domains}")
        return {"domains": sorted_domains, "count": len(sorted_domains)}

    except Exception as e:
        logger.error(f"[DOMAINS] Failed to retrieve domain list: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve domain list: {str(e)}")
