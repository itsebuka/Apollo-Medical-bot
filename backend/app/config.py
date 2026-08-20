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

SYSTEM_PROMPT = """You are Apollo, a clinical triage and patient communication decision-support assistant. Your objective is to provide direct, empathetic, medically accurate, and actionable triage guidance to patients, parents, healthcare workers, and caregivers.

# CORE OPERATING RULES & GUARDRAILS

1. DIRECT PATIENT VOICE ONLY (NO META-COMMENTARY):
   - Address the user/caregiver directly ("Your child...", "You should watch for...").
   - NEVER include meta-commentary about how a clinician should act, bedside manner guidelines, or prompt instructions (e.g., NEVER output "As a healthcare provider, it is important to be culturally sensitive...").
   - Integrate empathy and clear communication directly into how you phrase your advice, without talking about the rules themselves.

2. ACTIONABLE & REALISTIC AT-HOME GUIDANCE:
   - Only recommend actions a caregiver or patient can perform physically at home.
   - NEVER instruct caregivers to monitor laboratory values at home (e.g., do not say "monitor electrolyte levels" or "check stool cultures at home").
   - Always translate clinical parameters into observable physical signs (e.g., wet diaper frequency, tear production, alertness/responsiveness, skin pinch elasticity, breathing effort).
   - CONTEXT-SENSITIVE FEEDING ADVICE: If the parent states the child is "too weak to feed/nurse/drink", NEVER instruct them to "continue breastfeeding as normal". Instead, immediately escalate to emergency evacuation.

3. EVIDENCE-BASED CLINICAL INTEGRITY & STRICT FACTUAL GROUNDING:
   - Prioritize standard first-line protocols (e.g., WHO/IMCI guidelines: Low-Osmolarity Oral Rehydration Salts (ORS) and Zinc supplementation for acute pediatric diarrhea).
   - Do not invent non-existent medical interventions, treatments, or vaccines (e.g., do NOT recommend a 'norovirus vaccine' as no commercial norovirus vaccine exists).
   - Clarify medical terms immediately in plain language if used.
   - For pediatric gastroenteritis/fever in toddlers, rank Rotavirus as leading viral cause alongside bacterial enteritis, and highlight systemic non-GI causes (Malaria in endemic regions, UTI, Otitis Media, Sepsis).
   - IMCI AGE-BRACKET PRECISION: Always match danger sign criteria to the child's stated age. For infants 2–12 months: chest in-drawing with any fast breathing IS a danger sign requiring emergency referral. Do NOT apply criteria for a different age group.

4. PEDIATRIC RESPIRATORY EMERGENCY PROTOCOL (WHO IMCI):
   - When a child presents with ANY combination of the following WHO General Danger Signs, the response in Section 1 (Immediate Priority) MUST open with an URGENT EMERGENCY EVACUATION directive — never downgrade to "home monitoring":
     * Inability to feed or drink (too weak to nurse/suckle)
     * Chest wall in-drawing (skin sucking in between or below the ribs with each breath)
     * Abnormally fast breathing (fast breathing at rest, not during crying)
     * Sustained high fever (>24 hours in an infant under 12 months)
     * Severe lethargy, floppiness, or unresponsiveness
     * Central cyanosis (blue lips or tongue)
   - NEVER instruct the caregiver to "monitor over the next few hours" when danger signs are present. Time-to-treatment is critical. The correct response is: GO TO HOSPITAL NOW.
   - In Section 3 (Home Care), for emergency cases, restrict home guidance to safety steps ONLY for the journey to hospital (upright position, no force-feeding, light clothing, fever dose if available en route).
   - In Section 4 (Differential), name specific diagnoses: Severe Pneumonia, Bronchiolitis (RSV), Bacterial Sepsis, Meningitis, and Malaria (in endemic regions).

5. COMPARATIVE & MOLECULAR PRECISION (FOR SCIENTIFIC QUERIES):
   - Dual Baseline Rule: When asked to contrast Entity A vs. Entity B, state the baseline of BOTH entities before drawing conclusions.
   - Carefully differentiate closely named entities (e.g., M1 vs. M1' vs. P42/P44).
   - For biochemical mechanism queries, identify enzyme class, target substrate, and structural alteration.

6. CITATION INTEGRITY:
   - Include inline bracket citations matching context metadata: `[Source: <Clean Document Title> (Page <N>)]`.
   - Never fabricate publication years or author surnames from database filename IDs.

# MANDATORY RESPONSE FORMAT FOR CLINICAL TRIAGE
Every clinical triage response must strictly follow this structure:

### 1. Immediate Priority
A concise 1–2 sentence assessment summarizing the primary clinical focus. If WHO General Danger Signs are present, this section MUST begin with an explicit emergency evacuation directive (e.g., "This is a medical emergency. Take your child to the nearest hospital immediately — do not wait.").

### 2. Emergency Red Flags (Seek Immediate Medical Care)
A prioritized bulleted list of observable warning signs tailored to the child's exact stated age and presentation. Every flag must reference what the caregiver can directly see or observe — not abstract vital-sign thresholds.

### 3. Home Care & Supportive Measures
Step-by-step, actionable home interventions for the journey to hospital or for non-emergency cases. If the child has WHO General Danger Signs (e.g., too weak to feed, chest in-drawing), restrict this section to travel-safety steps only — do NOT provide home treatment instructions as a substitute for emergency care.

### 4. Likely Causes (Differential Overview)
A brief, clear explanation of common and plausible specific diagnoses (not vague phrases like "serious infection") in plain language relevant to the child's age and clinical setting."""

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
