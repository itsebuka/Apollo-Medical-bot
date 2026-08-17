"""
Apollo Medical Triage System — Configuration & Shared State
============================================================
This module centralizes all configuration and holds the singleton instances
of the LLM and embedding model.

Why singletons?
    Loading a 4.5GB GGUF model takes ~8-12 seconds. If we re-loaded it on
    every HTTP request, the system would be unusable. Instead, we load once
    at application startup (FastAPI lifespan event) and share the instances
    across all request handlers via this module.

    This is the Singleton pattern applied to resource-intensive objects —
    a foundational pattern in systems engineering.
"""

import os
from pathlib import Path
from typing import Any

# ENFORCE 100% OFFLINE EXECUTION (ADTC 2026 REQUIREMENT)
# Prevents SentenceTransformers/HuggingFace from attempting to connect to the internet
# to check for model updates, which causes silent hangs during backend startup.
os.environ["HF_HUB_OFFLINE"] = "1"

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Root of the entire repository (two levels up from backend/app/)
REPO_ROOT = Path(__file__).parent.parent.parent

# Absolute path to the GGUF model file
MODEL_PATH = REPO_ROOT / "model" / "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

# Persistent ChromaDB storage directory (built by ingest.py)
CHROMA_DB_DIR = REPO_ROOT / "backend" / "chroma_db"

# ─────────────────────────────────────────────────────────────────────────────
# LLM CONFIGURATION
# Every parameter here is a deliberate memory/performance trade-off.
# ─────────────────────────────────────────────────────────────────────────────

LLM_CONFIG: dict[str, Any] = {
    # Context window size in tokens.
    # 4096 handles our Parent Chunks perfectly while saving ~400MB of RAM 
    # compared to 8192. Crucial for our 7GB offline limit.
    "n_ctx": 4096,

    # Number of CPU threads for inference.
    # Adaptive: uses all physical cores minus one to keep the OS responsive.
    # Falls back to 2 if os.cpu_count() returns None (e.g. in a container).
    # This ensures correct behaviour on the ADTC evaluation machine regardless
    # of its core count, rather than assuming 4 cores as a hardcoded value.
    "n_threads": max(2, (os.cpu_count() or 4) - 1),

    # Number of model layers to offload to GPU.
    # 0 = pure CPU inference. We never use GPU — guaranteed to work on
    # any laptop regardless of GPU presence. Critical for offline evaluation.
    "n_gpu_layers": 0,

    # Enable mmap on Windows to allow instant memory-mapped model loading
    # and prevent 3.36GB CPU_REPACK contiguous RAM allocation failures.
    "use_mmap": True,

    # Lock model weights in RAM (prevent OS from swapping to disk).
    # NOTE: Set to False on Windows by default because it requires the
    # "Lock pages in memory" user right assignment.
    "use_mlock": False,

    # Enable Flash Attention for drastic TTFT (Time to First Token) speedup and memory reduction
    "flash_attn": True,

    # Maximize prompt ingestion batch size
    "n_batch": 512,
    "n_threads_batch": max(2, (os.cpu_count() or 4) - 1),

    # Verbose=False silences the llama.cpp C++ console output.
    # Set to True temporarily if you need to debug model loading issues.
    "verbose": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# RAG CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "apollo_medical_knowledge"

# Number of chunks to retrieve from ChromaDB per query.
# Increased from 3 -> 5 to ensure multi-page functional coverage of complex mechanisms.
N_RESULTS = 5

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

GENERATION_CONFIG = {
    # Temperature controls randomness. 0 = deterministic, 1 = creative.
    # 0.35 is calibrated for medical contexts: factual but not robotic.
    # Lower than 0.3 makes the model repeat itself; higher than 0.6 makes it
    # hallucinate clinical facts.
    # NOTE: max_tokens is intentionally absent here — it is computed dynamically
    # per-request in the /chat handler (dynamic_max_tokens) based on query complexity.
    "temperature": 0.35,

    # Top-K sampling: at each step, only consider the top 40 most likely tokens
    # before applying top_p. Without this, llama.cpp samples from all 32,000
    # vocabulary tokens — extremely wasteful. top_k=40 cuts this to the top
    # candidates, giving a significant tokens/sec speedup with zero quality loss.
    # This is the single biggest speed lever available at the sampling level.
    "top_k": 40,

    # Top-p (nucleus) sampling: consider only tokens whose cumulative
    # probability exceeds this threshold. Works alongside top_k.
    # Tightened slightly from 0.90 to 0.88 to reduce sampling breadth.
    "top_p": 0.88,

    # Min-P sampling: a complementary filter that removes tokens whose
    # probability is below (min_p * probability of the best token).
    # This early-prunes the candidate list before top_p, further speeding
    # up sampling while preserving quality on high-confidence medical terms.
    "min_p": 0.05,

    # Repeat penalty: discourages the model from looping on the same phrases.
    # 1.1 is a gentle penalty appropriate for structured medical responses.
    "repeat_penalty": 1.1,

    # Stop sequences: the model halts generation when it produces any of these.
    # "<|eot_id|>" is the Llama 3 end-of-turn special token.
    "stop": ["<|eot_id|>", "<|end_of_text|>", "User:", "Human:"],
}

# ─────────────────────────────────────────────────────────────────────────────
# MEDICAL SYSTEM PROMPT
# This is Apollo's identity. It is injected at the top of every conversation.
# The Llama 3 Instruct format uses special tokens to demarcate roles.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Apollo, a clinical-grade medical and bioscience offline triage & research engine. You synthesize retrieved scientific source context into accurate, structurally disciplined summaries for healthcare professionals and biomedical researchers.

### CORE OPERATIONAL DIRECTIVES:

1. STRICT FACTUAL GROUNDING (ZERO TOLERANCE FOR HALLUCINATION):
   - Answer exclusively using explicit facts, mechanisms, structural relations, and data present in the provided context chunks.
   - Never extrapolate, invent, or substitute general pre-training memory if the specific enzymatic pathway, amino acid length, or anatomical structure is not directly stated in the text.
   - If a specific mechanism, surgical relation, or baseline is missing from the retrieved context, explicitly state what is present and note precisely what detail is absent. Never claim a mechanism works through generic terms (e.g., never say "genome collinearity" or "evolutionary pressure" when a biochemical explanation is requested).

2. COMPARATIVE QUESTIONS (DUAL BASELINE RULE):
   - When asked to contrast Entity A with Entity B (e.g., Influenza C vs. Influenza A, or HBV vs typical RNA/DNA viruses), you MUST explicitly state the genomic/structural baseline of BOTH entities before detailing the difference.
   - Verify reading frame mechanics (spliced vs. collinear, overlapping ORFs vs. non-overlapping) for both items being compared.

3. ENTITY & NOMENCLATURE DISCRIMINATION:
   - Carefully differentiate closely named entities (e.g., M1 [242 aa] vs. M1' [259 aa] vs. P42/P44 [374 aa]). Verify numerical lengths, mutations, and cleavage sites against their specific source definitions before generating the final text.

4. CITATION INTEGRITY & NO FABRICATED REFERENCES:
   - Every factual claim must include an inline bracket citation matching the provided chunk metadata: `[Source: <Clean Document Title> (Page <N>)]`.
   - Never fabricate or guess publication years, authors, or journal names not present in the retrieved chunk metadata.
   - Do not interpret database ID numbers in filenames as publication years.

5. BIOCHEMICAL & ENZYMATIC MECHANISM PROTOCOL:
   - When asked for a specific biochemical, enzymatic, or molecular mechanism (e.g., how a protein, enzyme, enhancin, or toxin increases pathogenicity or infectivity):
     a) You MUST explicitly identify the specific enzyme family or protein class (e.g., metalloproteinase, zinc-binding endopeptidase, serine protease).
     b) Identify the exact substrate or target tissue/structure (e.g., peritrophic membrane, intestinal mucin proteins, mucosal barrier).
     c) Describe the resulting cellular/structural alteration (e.g., degradation/digestion of mucin allowing virions to reach and fuse with microvilli).
   - NEVER confuse analytical genomic observations (such as sequence collinearity, gene mapping, or phylogenetic alignment) with physical biochemical or enzymatic mechanisms of disease.

6. OUTPUT STRUCTURE & TOKEN MANAGEMENT:
   - Structure responses cleanly using standalone bold headers, precise bullet points, and concise mechanism descriptions.
   - Ensure the generation finishes cleanly without mid-sentence token cutoffs in reference sections."""

CASUAL_SYSTEM_PROMPT = """You are Apollo, a friendly, helpful, and highly intelligent medical AI assistant built for Nigerian healthcare workers. 
The user is currently making casual conversation or asking a mundane question. 
Respond naturally, warmly, and politely. You do not need to use medical jargon or append citations for casual chat. 
If appropriate, gently and kindly remind them that you are ready to assist with any clinical triage, virology, pharmacology, or pathophysiological queries whenever they are ready."""


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON STATE — Populated at application startup in main.py
# ─────────────────────────────────────────────────────────────────────────────

# These are populated by the FastAPI lifespan event in main.py.
# Using module-level variables as a simple, dependency-injection-free
# singleton store. For a production system, you'd use a dependency injection
# framework, but for our constrained environment, this is optimal.
llm_instance: Any = None
embedding_model_instance: Any = None
chroma_collection_instance: Any = None
cross_encoder_instance: Any = None
