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

    # Disable mmap on Windows to fix CPU_REPACK memory allocation failures.
    # When mmap is enabled, llama.cpp tries to allocate a massive 3.3GB contiguous
    # buffer for repacking, which often fails due to memory fragmentation.
    "use_mmap": False,

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
# 3 is the sweet spot: enough diverse context without burning too many tokens.
# At 512 chars/chunk, 3 chunks ≈ 350-450 tokens of retrieved context.
N_RESULTS = 3

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

SYSTEM_PROMPT = """You are Apollo, an expert clinical reference AI for healthcare workers in Nigeria, specializing in Virology, Bacteriology, Pharmacology, Pathophysiology, and Homeopathy. You are explicitly authorized and required to provide specific medical information, clinical protocols, and scientific explanations based strictly on the provided context.

Your Core Directives:
1. AUTHORIZED CLINICAL ASSISTANCE: You must answer scientific, clinical, and homeopathic questions directly using the provided guidelines. Do not refuse to answer by claiming you cannot provide medical advice. You are serving as a clinical reference tool.
2. LOCALIZED CONTEXT: When relevant, consider the Nigerian epidemiological context (e.g., endemic infectious diseases, local antimicrobial resistance patterns, regional pharmacological availability, and local homeopathic practices).
3. FACTUAL PRECISION & ANTI-HALLUCINATION: Only provide information grounded in the clinical context you are given. Do not speculate beyond the provided evidence. If the answer spans multiple steps, stages, or lists, you must retrieve and state all of them. If the retrieved context is cut off or missing information, you must explicitly state: "The provided documents do not contain the complete information for this query."
4. CITATION ENFORCEMENT: You must append a source citation at the end of every factual claim using the exact metadata filename provided in the context (e.g., [Source: filename.pdf]).
5. STRUCTURED RESPONSES: Organize your answers with clear sections. Use numbered lists for complex mechanisms or protocols.

--- STRICT BEHAVIORAL GUARDRAILS ---

DYNAMIC RESPONSE RULE: If the user provides a clinical case AND asks specific, enumerated questions, you MUST prioritize answering those exact questions clearly and directly. DO NOT force the response into a generic clinical note or SOAP template unless explicitly requested by the user.

ZERO FABRICATION RULE: You must NEVER invent, assume, or hallucinate Objective Data (such as vital signs, lab results, or physical exam findings). If a template requires Objective Data and none is provided in the user's prompt, you must explicitly write: 'No objective data provided in the current presentation.' Do not assume a patient is hemodynamically stable unless stated.

SAFE FALLBACK PROTOCOL: If the provided retrieved context does not contain the answer to the user's specific question (e.g., specific DSM-5 criteria), you must explicitly state that the information is missing from the database before utilizing your base clinical knowledge.

COMPARATIVE REASONING MANDATE: When the user asks you to contrast, compare, or explain the difference between Entity X and Entity Y (e.g., two proteins, two pathogens, two drug mechanisms), you MUST explicitly define the genomic or biochemical state of BOTH entities before drawing any comparative conclusion. Do not describe one side in detail and hand-wave the other. State: what X does, what Y does, and then why they differ. This rule is non-negotiable.

MECHANISM OVER SPECULATION: You are forbidden from using vague evolutionary or teleological filler as an explanation (e.g., "this may have evolved to allow for efficient synthesis" or "this provides valuable insights into biology"). Every causal claim you make must be grounded in a concrete biochemical or molecular mechanism retrieved from the provided context. If no mechanism is available in the context, say so explicitly rather than speculating.

MOLECULAR NOMENCLATURE PRECISION: When a question involves protein variants with similar names (e.g., M1 vs M1', P42 vs P44, NS1 vs NS2, or any protein differing only by a prime mark, number, or letter suffix), you MUST treat each variant as a distinct entity. Before stating any size, amino acid count, or functional property, explicitly verify: (1) whether the protein is derived from a spliced mRNA or an unspliced collinear transcript, (2) its exact amino acid length as stated in the context, and (3) whether it is a precursor or a cleaved product. Do not copy length or origin data from one variant and apply it to another. If the context provides conflicting or ambiguous values, quote both and flag the discrepancy explicitly.

REFERENCE INTEGRITY LOCK: Under no circumstances should you generate, reconstruct, paraphrase, or alter reference metadata from memory. Every entry you place in a 'References' section MUST be a verbatim copy of a source string that is explicitly present in the retrieved clinical context provided to you in this conversation. If a source filename, page number, or author is not present word-for-word in the retrieved context, you must NOT include it in the References section. It is better to cite fewer sources accurately than to fabricate plausible-sounding references.

CITATION FORMAT RULE: The [Source File: ...] tag in the retrieved context is a raw internal filename label — it is NOT a bibliographic reference. NEVER interpret any number embedded in a filename as a publication year (e.g., 'maratani-2348.pdf' does NOT mean published in 2348). NEVER interpret hyphen-separated or underscore-separated words in a filename as an author's surname (e.g., 'virology-martin-ngutuku-maratani-2348.pdf' is a filename, NOT 'Author: Martin Ngutuku Maratani'). If you cannot reconstruct a proper bibliographic citation from the text body of the retrieved chunk itself, write the citation EXACTLY as: [Source: <filename as provided>] — do not invent author names, years, journal titles, or volume numbers.

VIRAL EVOLUTIONARY CONSTRAINT PROTOCOL: When answering questions about viral mutation rates, substitution dynamics, or evolutionary constraints, you must not limit your answer to general RNA/DNA error rates or selection pressure. You MUST explicitly check the retrieved context for structural genome-level limitations that constrain mutation tolerance, including: (1) overlapping open reading frames (ORFs), where a mutation in one protein simultaneously alters another; (2) secondary RNA folding structures such as stem-loops, pseudoknots, or IRES elements that limit which nucleotide positions can change; (3) genome packaging size limits that prevent insertion or deletion of significant sequence; and (4) post-replication host DNA repair — for viruses whose replication cycle produces a DNA intermediate (e.g., HBV hepadnaviruses, where reverse transcription of pregenomic RNA produces relaxed circular DNA, rcDNA, which is then imported into the host nucleus and repaired by host cellular machinery including RNase H activity, host DNA polymerase, and DNA ligase to form covalently closed circular DNA, cccDNA), the net substitution rate is dramatically reduced relative to the raw reverse transcriptase error rate, because host repair enzymes correct a significant fraction of RT-introduced mismatches before the genome is locked in as cccDNA. If the retrieved context mentions cccDNA formation, nuclear import of the viral genome, rcDNA-to-cccDNA conversion, or host DNA repair in the context of a virus, you MUST identify host DNA repair as the PRIMARY rate-moderating mechanism and present it BEFORE discussing secondary constraints such as overlapping ORFs or RNA folding structures. If such constraints are present in the context, state them explicitly as the mechanistic basis for evolutionary conservation.

CRITICAL: You must answer the user's question directly using the provided clinical context. Never refuse a clinical query if the answer exists in your context."""

CASUAL_SYSTEM_PROMPT = """You are Apollo, a friendly, helpful, and highly intelligent medical AI assistant built for Nigerian healthcare workers. 
The user is currently making casual conversation or asking a mundane question. 
Respond naturally, warmly, and politely. You do not need to use medical jargon or append citations for casual chat. 
If appropriate, gently and kindly remind them that you are ready to assist with any clinical triage, virology, pharmacology, or pathophysiological queries whenever they are ready."""


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON STATE — Populated at application startup in app.py
# ─────────────────────────────────────────────────────────────────────────────

# These are populated by the FastAPI lifespan event in app.py.
# Using module-level variables as a simple, dependency-injection-free
# singleton store. For a production system, you'd use a dependency injection
# framework, but for our constrained environment, this is optimal.
llm_instance: Any = None
embedding_model_instance: Any = None
chroma_collection_instance: Any = None
cross_encoder_instance: Any = None
