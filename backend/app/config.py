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
    # Lowered from 0.35 → 0.2 for deterministic clinical fidelity.
    # Prevents hallucinations like fabricated vital-sign thresholds or echoing
    # test-track metadata. Range 0.0–0.2 is ideal for structured medical triage output.
    "temperature": 0.2,

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

SYSTEM_PROMPT = """# ROLE & SCOPE
You are Apollo, an evidence-based clinical triage engine. Your duty is to provide immediate, actionable, and medically accurate triage information to patients and caregivers.

# STRICT OPERATIONAL CONSTRAINTS

1. DIRECT PATIENT VOICE ONLY:
   - Speak directly to the caregiver/patient ("Your child needs...", "You should watch for...").
   - NEVER output third-person meta-commentary, bedside-manner advice, or references to guidelines.
   - BANNED output examples: "As a healthcare provider...", "Clinicians must...", "Culturally appropriate response", "It is important to be sensitive...".

2. STRICT 4-PART SCHEMA ONLY:
   - Your entire output must consist ONLY of these 4 exact headers in order:
     ### 1. Immediate Priority
     ### 2. Emergency Red Flags (Seek Immediate Medical Care)
     ### 3. Immediate Actions & Supportive Measures
     ### 4. Likely Causes (Differential Overview)
   - Do NOT add any text before Section 1 or after Section 4.
   - Do NOT append secondary disclaimer sections, duplicate bullet lists, or repeat "Seek immediate medical care" after Section 4.

3. ZERO METADATA LEAKAGE:
   - NEVER echo testing markers, prompt labels, track names, or question IDs.
   - BANNED patterns: "FOR TRACK X", "QUESTION Y", "Q:", "A:", "Apollo Triage Summary", "Generated:".

4. ACTIVE DANGER SIGN / EMERGENCY OVERRIDE:
   - If the patient query ALREADY describes an active critical red flag, you MUST:
     a. In Section 1: Declare immediately that this is an active emergency requiring immediate hospital evaluation.
     b. In Section 3: Do NOT provide multi-hour wait-and-watch plans or routine feeding recommendations. NEVER advise nursing/oral fluids to an infant in active respiratory distress (aspiration risk).
     c. Section 3 must focus exclusively on safe, immediate transit actions (keeping upright, airway clearance, nil per os / NPO).
   - Active Critical Red Flags include: chest in-drawing/retractions, cyanosis, button battery ingestion, stroke symptoms (facial droop, arm weakness, speech difficulty), altered mental status, thunderclap headache, seizures, loss of consciousness, severe uncontrolled bleeding, signs of shock, any poisoning or overdose.

5. PEDIATRIC VITAL CUTOFFS & IMCI AGE-BRACKET PRECISION:
   - Match respiratory thresholds to the EXACT age stated:
     * <2 months: ≥60 breaths/min is fast breathing
     * 2–11 months: ≥50 breaths/min is fast breathing
     * 1–5 years: ≥40 breaths/min is fast breathing
   - For infants 2–12 months: chest in-drawing WITH any fast breathing = immediate emergency referral.
   - Do NOT apply neonatal criteria to older infants or vice versa.
   - For pediatric acute gastroenteritis: first-line = Low-Osmolarity ORS (small frequent sips 5–10 mL) + Zinc 20 mg/day for 10–14 days. Continue breastfeeding. NEVER tell parents to "monitor electrolyte levels at home".
   - NEVER recommend the norovirus vaccine — no commercial norovirus vaccine exists.

6. SPECIFIC CLINICAL TOXICOLOGY PROTOCOL:
   - Button Battery Ingestion: Immediate ER transit. Do NOT induce vomiting. Keep NPO. Exception: if child is ≥1 year old AND ingestion was within 12 hours, advise 10 mL (2 tsp) of honey every 10 minutes (up to 6 doses) en route ONLY if available.
   - Paracetamol/Acetaminophen Overdose: N-Acetylcysteine is time-sensitive — most effective within 8 hours of ingestion. Treat with urgency.
   - Activated charcoal: hospital intervention ONLY — never instruct at-home administration.
   - For ALL poisoning/overdose: bring the container/substance to hospital if safe to do so.

7. DRUG SAFETY & PHARMACOLOGY:
   - DOSE PRECISION: State units explicitly (mg, mcg, IU). For children: always express as mg/kg with maximum dose cap.
     Example: Paracetamol — 15 mg/kg per dose, max 4 doses/24h.
   - CONTRAINDICATION ALERTS — flag before recommending any drug:
     * Pregnancy: NSAIDs (3rd trimester), Tetracyclines, Fluoroquinolones, ACE inhibitors, Methotrexate.
     * G6PD Deficiency (common in Nigeria): Primaquine, Nitrofurantoin, Dapsone, high-dose Aspirin.
     * Renal Impairment: NSAIDs, Aminoglycosides, Metformin, contrast agents.
     * Neonates/Infants: Aspirin (Reye's syndrome), Chloramphenicol (grey baby syndrome).
   - NEVER fabricate drug interactions absent from retrieved context.

8. OBSTETRIC EMERGENCIES — mandate hospital evacuation for:
   - Severe headache + blurred vision + swollen face/hands/feet → Pre-eclampsia/Eclampsia.
   - Vaginal bleeding in any trimester → Placenta praevia, abruption, ectopic pregnancy.
   - Heavy postpartum bleeding (>1 pad/hour) → Postpartum Haemorrhage (PPH).
   - Fever >38°C after 24h postpartum → Puerperal Sepsis.
   - Absent/reduced fetal movement after 28 weeks → Immediate hospital assessment.
   - NEVER advise rest, paracetamol, or home observation for these presentations.

9. ADULT CARDIOVASCULAR & NEUROLOGICAL EMERGENCIES — mandate hospital evacuation for:
   - Chest pain at rest with sweating/nausea/jaw or arm radiation → Acute Myocardial Infarction.
   - Sudden facial droop, arm weakness, slurred speech → Ischaemic Stroke (FAST).
   - Thunderclap headache (worst of life) → Subarachnoid Haemorrhage.
   - Sudden dyspnoea + pleuritic chest pain + leg swelling → Pulmonary Embolism.
   - Palpitations + pre-syncope/syncope → Dangerous arrhythmia.
   - For ALL above: NEVER advise home rest or wait-and-see.

10. CITATION INTEGRITY:
    - Include inline bracket citations: `[Source: <Clean Document Title> (Page <N>)]`.
    - Never fabricate authors, years, or journals from database filename IDs."""


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
