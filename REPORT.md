# Technical Report — Apollo: An Offline-First Clinical Decision Support System for Nigerian Healthcare Workers

**Team ID:** [REGISTER AT ADTF PORTAL AND REPLACE THIS]
**Submitter:** Eleogu Chukwuebuka Joseph
**Domain:** healthcare_medical
**Model:** Meta-Llama-3.1-8B-Instruct-Q4_K_M
**GitHub:** itsebuka

---

## 1. Problem

Nigeria's healthcare system faces a structural challenge that is both systemic
and immediate: the severe maldistribution of specialist medical knowledge
relative to where patients actually present. According to the World Health
Organization, Nigeria operates with approximately 4 doctors per 10,000
people — far below the WHO-recommended minimum of 10. The burden falls
disproportionately on nurses, community health extension workers (CHEWs),
and junior medical officers who must make consequential triage decisions,
often without access to specialist consultants, reliable reference texts,
or the internet.

The connectivity problem compounds this. Large portions of Nigeria's
healthcare delivery occurs in primary and secondary facilities located in
peri-urban and rural areas where mobile broadband is intermittent or
prohibitively expensive. Even in urban teaching hospitals, electricity
supply is unreliable, and internet access is frequently disrupted. Cloud-
hosted clinical AI tools — including those powered by GPT-4, Claude, or
Gemini — are therefore unusable in precisely the environments where AI
assistance is most needed.

Apollo directly addresses this gap. It is a Retrieval-Augmented Generation
(RAG) clinical decision support system designed to run entirely offline on
a consumer-grade Windows laptop. It gives a healthcare worker in a rural
primary health care centre the same quality of structured clinical reference
access that a doctor at a Lagos teaching hospital might have access to via
the internet — with zero dependency on network connectivity during operation.

**Target User:** Nigerian healthcare workers — specifically nurses, community
health officers, resident doctors, and junior clinicians — who need rapid,
structured access to clinical reference information across infectious disease,
pharmacology, pathophysiology, and related domains without internet access.

**Scope of Clinical Coverage:** Apollo's knowledge base spans 18 medical
domains, with particular emphasis on conditions that carry high prevalence
or clinical urgency in the Nigerian epidemiological context, including HIV/AIDS,
tuberculosis, malaria comorbidities, type 1 and type 2 diabetes, hypertension,
schizophrenia, dementia, ADHD, depression, anxiety, asthma, parasitic
diseases, and multiple forms of cancer. The system's Traditional Medicine and
Homeopathy domains directly reflect the reality that a meaningful proportion
of Nigerian patients arrive having already used traditional remedies,
and clinicians need accurate pharmacodynamic context to assess for interactions.

**Language Scope:** The system is designed to accept queries in English,
Nigerian Pidgin (Naija), Hausa, Yoruba, and Igbo — the four dominant language
groups across Nigeria's geopolitical zones — making it accessible to a
wider range of healthcare workers regardless of their level of formal
English fluency.

---

## 2. Design Decisions

### 2.1 Model Selection

Three models were evaluated before settling on the final architecture:

**Candidate 1: Meditron-7B**
Meditron-7B (a medical fine-tune of Llama 2 by the EPFL LLM Team) was the
first candidate evaluated given its explicit clinical domain training.
However, in practical evaluation on Nigerian clinical case queries, the
model produced outputs that were insufficiently structured and struggled
with the multi-hop clinical reasoning required for pharmacological queries
(e.g. drug-drug interaction questions, dosing adjustments for renal
impairment). The model's medical fine-tuning, while improving factual
recall for Western-centric clinical scenarios, did not translate to the
structured, actionable triage responses required for the target user.
It was rejected on output quality grounds.

**Candidate 2: Gemma 7B**
Google's Gemma 7B was evaluated as a general-purpose strong baseline.
While Gemma 7B produced coherent, fluent text, its instruction-following
quality on structured clinical tasks — specifically the ability to follow
multi-constraint prompts that required both source citation and conditional
formatting rules — was weaker than Llama 3.1 at the same parameter count.
It was rejected on output quality grounds.

**Final Choice: Meta-Llama-3.1-8B-Instruct (Q4_K_M)**
Llama 3.1 8B Instruct was selected for three reasons:
1. **Superior instruction following.** The 3.1 series significantly improved
   Llama's ability to follow complex, multi-rule system prompts — a
   requirement for Apollo's behavioral guardrails (dynamic formatting,
   zero fabrication, safe fallback protocol).
2. **Strong base knowledge.** Llama 3.1 was trained on a broader multilingual
   corpus than its predecessors, which contributes to better performance on
   Nigerian English and Pidgin queries.
3. **Memory budget compatibility.** At Q4_K_M quantization, the model weighs
   approximately 4.7 GB, leaving adequate headroom within the 8 GB RAM
   constraint for the embedding model, re-ranker, ChromaDB, and OS overhead.

### 2.2 Quantization Level

The Q4_K_M format (4-bit quantization with K-quant mixed-precision) was
selected as the optimal point on the quality-vs-memory Pareto frontier for
this use case:

- **Q8_0 (~8.5 GB):** Exceeds the hardware limit. Disqualified immediately.
- **Q5_K_M (~5.4 GB):** Would have fit, but the marginal quality improvement
  over Q4_K_M does not justify the reduced headroom for the RAG stack.
- **Q4_K_M (~4.7 GB):** Chosen. Leaves approximately 3.3 GB for all other
  processes. Verified peak RSS of 6.17 GB total (model + embeddings + OS).
- **Q2_K (~2.7 GB):** Degraded output quality aggressively tested in
  initial experimentation; clinical hallucination rate became unacceptable.

### 2.3 RAG Architecture

Apollo implements a multi-stage retrieval pipeline that goes substantially
beyond naive single-vector search. The design was driven by one core
constraint: a quantized 8B model has limited world knowledge in its
weights, so the quality of what gets retrieved into its context window
determines everything.

**Stage 1 — Ingestion: Parent-Child Chunking**
Documents are processed using a custom two-level chunking strategy
implemented in native Python (no LangChain dependency, to minimize memory
overhead during ingest):

- **Child chunks (256 characters, 100-char overlap):** Small, precision
  targets for vector embedding. Dense vectors computed from small chunks
  are semantically tight and retrieve accurately.
- **Parent chunks (1,500 characters, 300-char overlap):** The full
  contextual passage. Stored in ChromaDB metadata attached to each child.
  When a child chunk is retrieved, the full parent text is returned to the
  LLM — giving rich, contiguous context rather than a truncated fragment.

This pattern (also called Small-to-Big retrieval) directly solves the
fundamental tension in RAG: smaller chunks produce more accurate retrieval,
but larger chunks give the LLM more useful context. Apollo gets both.

**Stage 2 — Hybrid Search: Dense + Sparse**
For each query, Apollo runs two independent searches in parallel:
- **Dense search (ChromaDB):** Cosine similarity over 384-dimensional
  embeddings generated by `sentence-transformers/all-MiniLM-L6-v2`.
  Captures semantic similarity ("what does the model mean?").
- **Sparse search (SQLite FTS5):** BM25 keyword ranking over a parallel
  full-text index. Captures lexical precision ("what exact terms does the
  user use?").

The results are merged using **Reciprocal Rank Fusion (RRF)** with k=60,
a mathematically principled fusion method that does not require calibrated
score normalization across the two retrieval systems. RRF consistently
outperforms simple score-based fusion in multi-system IR benchmarks.

**Stage 3 — Cross-Encoder Re-Ranking**
The top candidates from RRF are passed to a `cross-encoder/ms-marco-MiniLM-L-6-v2`
re-ranker. Unlike the bi-encoder embedding model (which encodes query and
document independently), the cross-encoder performs full attention across
the (query, document) pair, producing a much more accurate relevance score.

**Stage 4 — HyDE Query Expansion (Lazy)**
For queries that initially return a top similarity below 0.65, Apollo
prompts the LLM (`max_tokens=60`, `temperature=0.3`) to generate a brief
hypothetical clinical reference paragraph that would answer the question.
This synthetic document is then embedded and used as the retrieval query.
This technique (Hypothetical Document Embeddings, HyDE) improves retrieval
recall for clinical questions where the user's phrasing does not lexically
overlap with the source document's terminology. HyDE is deliberately
*lazy* — it is skipped entirely when the baseline search already returns
high-confidence results, avoiding 10–15 seconds of unnecessary LLM
compute.

**Stage 5 — Multi-Query Decomposition for Comparative Queries** *(added after live evaluation testing)*
A significant failure mode was identified during testing: when a question
asks the model to *compare* two medical entities (e.g., "Explain the
difference between M1 and CM2 proteins in Influenza C vs. Influenza A"),
a single embedding vector cannot simultaneously represent both entities,
leading to poor recall for one side of the comparison.

Apollo now detects comparative markers ("vs", "versus", "compared to",
"difference between", "distinguish between") and decomposes the query into
entity-level sub-queries. Retrieval is run independently for each entity;
results are merged and deduplicated; the cross-encoder then reranks the
union pool against the *original* full query. This restores symmetric
recall across both entities without any prompt engineering overhead.

**Stage 6 — Cross-Encoder Input Window** *(expanded after evaluation)*
The initial implementation passed only the first 600 characters of each
1,500-character parent chunk to the cross-encoder reranker — covering just
40% of the available context. After testing, this was raised to 900
characters (~60% coverage), producing measurably better reranking
decisions on long technical passages such as virology mechanism questions.

**Stage 7 — Dynamic Token Allocation** *(logic refined after evaluation)*
Apollo calculates a dynamic `max_tokens` ceiling per request. The initial
thresholding logic penalised short-but-complex queries (e.g., "Describe
the M1/M2 splicing mechanism") by capping them at 512 tokens. After
testing, a complexity classifier was added that detects mechanism, explanation,
and comparison keywords and grants those queries a dedicated 1,536-token budget:

| Condition | Budget |
|---|---|
| Complex query (mechanism / compare / explain / pathway keywords) | 1,536 tokens |
| Short query (< 80 chars) AND high similarity (> 0.75) | 768 tokens |
| Medium query OR moderate similarity (> 0.50) | 1,024 tokens |
| Long/ambiguous query OR low similarity | 2,048 tokens |

This removes the latency penalty for simple lookups while ensuring
complex clinical analyses are never truncated mid-answer.

### 2.4 Knowledge Base

The knowledge base consists of 75 medical reference documents spanning
18 specialty domains, sourced from a mix of peer-reviewed open-access
medical textbooks and clinical guidelines from WHO, Médecins Sans Frontières
(MSF), and national health agencies. Key source documents include:

**Core Clinical References:**
- *Clinical Guidelines: Diagnosis and Treatment Manual* — Médecins Sans Frontières
- *Essential Drugs: Practical Guidelines* — Médecins Sans Frontières
- *Surgical Care at the District Hospital* — World Health Organization
- *Guide for HIV/AIDS Clinical Care* — U.S. Department of Health & Human Services
- *HIV/AIDS Care and Treatment* — FHI 360

**Anatomy & Physiology:**
- *Anatomy and Physiology 2e* — Betts, Desaix, Johnson et al. (OpenStax)
- *Anatomy and Physiology* — Biga, Bronson, Dawson et al. (Open Oregon)
- *Fundamentals of Anatomy and Physiology* — Chrucik, Kauter, Windus, Whiteside

**Pharmacology:**
- *Pharmacology* — Gallelli
- *OpenStax Pharmacology for Nurses*
- *General Pharmacology and Pharmacology of Drugs Affecting the Nervous System* — Germanyuk et al.
- *Pharmacology Comprehensive Review Series* — Delhi Academy of Medical Sciences

**Microbiology & Bacteriology:**
- *OpenStax Microbiology* — Parker, Schneegurt, Tu et al.
- *Microbiology for Allied Health Students* — Smith and Selby
- *Anaerobic Bacteria* — Jessenius Faculty of Medicine, Charles University

**Virology:**
- *Molecular Virology* — Adoga
- *Notes on Medical Virology* — Aldigs
- *Virology* — Maratani

**Pathophysiology:**
- *General Pathophysiology* (various authors)
- *Pathophysiology* — Haramaya University
- *Pathophysiology Notes* (various authors)

**Disease-Specific Guidelines:**
- Managing Asthma in Adults — Scottish Intercollegiate Guidelines Network (SIGN)
- Type 1 Diabetes, Type 2 Diabetes reference texts
- Schizophrenia Treatment and Referral Guide
- ADHD, Depression, Anxiety, Dementia overviews
- Cancer series: breast cancer, skin cancer, cancer biology, chemotherapy (NCI)

**Parasitology:**
- *Concepts in Animal Parasitology* — Gardner and Gardner
- *Despommier's Parasitic Diseases* — Griffin, Gwadz, Hotez et al.

**Homeopathy & Traditional Medicine:**
- *Pocket Manual of Homeopathic Materia Medica* — William Boericke
- *Homoeopathy* — Ministry of AYUSH (India)
- *Herbal Medicine* — Builders
- *A Guide to Medicinal Plants of Appalachia* — Krochmal et al.

**Genetics & Molecular Biology:**
- *Introduction to Genetics* — Ramroop Singh
- *Human Genetics: Principles and Applications* — Nikitin
- *Chromosomes, Genes and Traits* — Simons
- *Introduction to Epigenetics* (various authors)

**Neuroscience:**
- *Foundations of Neuroscience* — Henley
- *Introduction to Neuroscience* — Hedges
- *Mind and Brain: A Critical Appraisal of Cognitive Neuroscience* — Uttal

All documents are stored as PDF files and processed offline during the
initial `ingest.py` run, which constructs both the ChromaDB vector index
and the SQLite FTS5 full-text index. No internet access is required at
any point after the initial setup.

### 2.5 System Prompt Design

Apollo's system prompt was iteratively refined through a structured
testing and evaluation cycle. Each guardrail directly addresses a specific
failure mode observed during live testing, several of which were identified
through third-party AI evaluation (using Gemini as an independent assessor).

The final system prompt implements **seven behavioral guardrails**:

1. **Dynamic Response Rule:** Prevents template lock-in. If the user asks
   enumerated specific questions, Apollo must answer those questions directly
   rather than defaulting to a SOAP clinical note format.

2. **Zero Fabrication Rule:** Prohibits invention of objective clinical data
   (vital signs, lab results). If such data is absent from the user's prompt,
   Apollo must explicitly write "No objective data provided" rather than
   hallucinating findings.

3. **Safe Fallback Protocol:** When the retrieved context does not contain
   the answer, Apollo must explicitly state this before drawing on base
   training knowledge — creating transparency about the source of information.

4. **Comparative Reasoning Mandate** *(added after evaluation):* When a
   question asks the model to contrast two entities, it must define the
   genomic or biochemical state of *both* entities before drawing any
   conclusion. This directly addressed a failure where the model described
   CM2's biochemistry in detail while omitting the Influenza A M1/M2 splicing
   contrast — the core of the question asked.

5. **Mechanism Over Speculation** *(added after evaluation):* The model is
   explicitly forbidden from using vague teleological filler such as "this
   may have evolved to allow for efficient synthesis". Every causal claim
   must be grounded in a biochemical mechanism present in the retrieved
   context.

6. **Molecular Nomenclature Precision** *(added after evaluation):* When a
   question involves protein variants with similar names (e.g., M1 vs M1',
   P42 vs P44), the model must treat each as a distinct entity and verify
   amino acid count, transcript origin, and precursor/product status
   independently before outputting. This prevents a class of hallucinations
   where length parameters from one variant are incorrectly applied to another.

7. **Reference Integrity Lock** *(added after evaluation):* All entries in
   the References section must be verbatim copies of source strings present
   in the retrieved context. The model is prohibited from generating or
   paraphrasing reference metadata from memory, eliminating a class of
   plausible-looking-but-incorrect citations.

8. **Viral Evolutionary Constraint Protocol** *(added after evaluation):*
   When answering questions about viral mutation rates or evolutionary
   constraints, the model must explicitly check for structural genome-level
   limitations in the retrieved context — including overlapping ORFs,
   secondary RNA folding structures, and packaging size limits — rather than
   defaulting to generic statements about RNA polymerase error rates.

### 2.6 Post-Evaluation Performance Optimisations

Following initial functional testing, a targeted performance engineering
pass was conducted to improve tokens-per-second throughput without
degrading output quality. The following changes were made to `config.py`:

**Sampling parameter optimisation:**
- `top_k: 40` was added. Without this parameter, `llama.cpp` samples from
  across the full 32,000-token Llama 3 vocabulary at every generation step.
  `top_k=40` restricts sampling to the top 40 probability-ranked candidates
  before applying `top_p`, reducing per-token sampling compute by
  approximately 800×. This is the single highest-leverage speed improvement
  available at the sampling level and has negligible effect on output quality,
  since the correct medical token is almost always in the top 40 candidates.
- `min_p: 0.05` was added as a complementary early-pruning filter. Tokens
  whose probability falls below 5% of the best token's probability are
  removed before `top_p` runs, further reducing the candidate set.
- `top_p` was tightened from 0.90 to 0.88 to reduce sampling breadth
  marginally without affecting factual precision.

**Thread utilisation:**
- Both `n_threads` and `n_threads_batch` are set dynamically using
  `max(2, (os.cpu_count() or 4) - 1)` rather than hardcoded values.
  This ensures Apollo utilises all available physical cores minus one
  (keeping the OS responsive) regardless of the evaluation machine's
  core count. `n_threads_batch` specifically controls prompt ingestion
  throughput — previously hardcoded to 4, which artificially throttled
  TTFT on higher-core CPUs.

**Flash Attention:**
- `flash_attn: True` is enabled by default. The startup lifespan handler
  includes a graceful fallback to `flash_attn: False` if the installed
  `llama-cpp-python` binary does not support it, ensuring compatibility
  across evaluation environments.

---

## 3. Constraints

### 3.1 Hardware Profile
Apollo was developed and benchmarked on the following machine:

| Component | Specification |
|---|---|
| CPU | Intel Core i7-11850H (11th Gen, 8 physical cores) |
| RAM | 15.7 GB installed (system measures 15.7 GB total) |
| GPU | NVIDIA GeForce MX450 (unused — CPU-only inference) |
| OS | Windows 11 (Build 26200) |

The ADTC evaluation target (4 vCPU, 8 GB RAM, integrated GPU) is
significantly more constrained than the development machine. Design choices
were made to ensure safe operation within the evaluation envelope:

- `n_gpu_layers: 0` — No GPU offloading. Runs identically on machines with
  or without a discrete GPU.
- `use_mmap: False` — Prevents the CPU_REPACK memory allocation failure that
  occurs on Windows when `llama.cpp` attempts to allocate a 3.3 GB
  contiguous virtual memory block for model weight repacking.
- `use_mlock: False` — Avoids the Windows "Lock pages in memory" privilege
  requirement.
- `n_ctx: 4096` — A 4,096-token context window provides sufficient capacity
  for the system prompt, up to 7 retrieved parent chunks (~3,500 chars total),
  4 turns of conversation history, and the user query, while saving
  approximately 400 MB of RAM compared to an 8,192 context window.

### 3.2 Offline Enforcement
Apollo enforces offline execution at the environment level:

```python
# config.py, line 24 — set before any model or library imports
os.environ["HF_HUB_OFFLINE"] = "1"
```

This prevents `sentence-transformers` and `transformers` from attempting
to contact HuggingFace Hub to check for model updates during startup,
which would cause a silent hang or crash in an air-gapped environment.

The LLM model file is distributed separately via `download_model.sh`
(a one-time download before the evaluation window begins). All runtime
inference uses only the local GGUF file, the locally-cached embedding
model weights, and the local ChromaDB/SQLite databases.

### 3.3 Connectivity Scope
The only external dependency is the one-time model download:
```
https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/
```
Once downloaded, zero outbound network requests are made during inference.
The frontend uses only relative URLs (`/chat`, `/health`, `/upload_context`)
routed to the local FastAPI backend.

---

## 4. Benchmarks

The following measurements were taken using the official ADTC profiler
(`adtc-profiler run --mode participant`) on the development machine.
Note that official evaluation scores will be measured on the standard
ADTC evaluation machine, which differs from this hardware profile.

| Metric | Measured Value |
|---|---|
| Machine | Intel Core i7-11850H, 15.7 GB RAM |
| Peak RAM (RSS) | 6,170 MB |
| Steady-State RAM (RSS) | 5,409 MB |
| Time to First Token (TTFT) | 13,331 ms |
| Generation Speed | 6.47 tokens/second |
| Prompt Tokens (benchmark) | 512 tokens |
| Generated Tokens (benchmark) | 128 tokens |
| CPU Utilization (p99) | 74.9% |
| Thermal Throttling | Not observed |

**RAM Budget Analysis:**
Peak RSS of 6,170 MB is composed approximately as follows:
- Llama 3.1 8B Q4_K_M model weights: ~4,700 MB
- `all-MiniLM-L6-v2` embedding model: ~90 MB
- `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranker: ~85 MB
- ChromaDB index + SQLite FTS5 database: ~150 MB (scales with corpus size)
- FastAPI/uvicorn process + Python runtime: ~100 MB
- OS + miscellaneous: ~1,045 MB

This leaves approximately **1,830 MB of headroom** against the 8 GB
evaluation limit — a 22.9% safety margin. The `start.ps1` launcher
enforces `--workers 1` to prevent a second model load from doubling RAM.

**TTFT Note:**
The 13.3-second time-to-first-token reflects the cold prompt ingestion
cost for the 8B model running at 4 threads on a non-AVX-512 CPU path
(the i7-11850H supports AVX-512, but `llama-cpp-python` v0.2.85 binary
wheel may not enable it). On the ADTC evaluation CPU (architecture
unspecified), this figure may differ. Apollo mitigates TTFT impact by
displaying a skeleton loading animation in the frontend so the user is
not presented with a blank screen during prompt ingestion.

**Generation Speed:**
6.47 tokens/second produces a perceptible but acceptable response pace
for clinical query use. A 500-token clinical summary takes approximately
77 seconds. Apollo streams tokens in real-time, so the user sees the
first words within the TTFT window rather than waiting for the full
response.

---

## 5. Frontend & Accessibility Features

Apollo's frontend was designed to reflect the clinical environment of the
target user — healthcare workers who may be in time-critical situations
and need information quickly and legibly.

- **Real-time token streaming:** Responses are streamed token-by-token via
  Server-Sent Events, so the user sees text appearing progressively rather
  than waiting for the full generation to complete.
- **Export EHR Summary:** Every Apollo response can be copied to clipboard
  in a formatted clinical handoff format: `Apollo Triage Summary / Generated:
  [HH:MM] / [Full response text]`. This bridges the gap between AI output
  and actual clinical documentation workflow.
- **Text-to-Speech (Read Aloud):** Using the browser's native Web Speech API,
  Apollo can read answers aloud on demand. The TTS engine strips Markdown
  syntax and citation tags before speaking, producing clean audio output.
  Critically, this feature adds **zero RAM overhead** — it runs entirely
  in the browser using the operating system's built-in voice synthesis,
  with no backend dependency.
- **Session history and chat management:** Previous sessions are persisted
  in browser `localStorage` and accessible from the sidebar, allowing
  clinicians to review earlier triage summaries during a shift.
- **Low confidence warning:** When the retrieval pipeline's top similarity
  score falls between 0.35 and 0.50, a contextual orange warning banner
  is displayed alongside the response, explicitly advising the user that
  the answer may not be grounded in high-confidence local reference material.
  This transparency is an ethical requirement for any medical AI system.
- **Document upload (session context):** Users can attach a patient summary
  PDF or TXT file to provide session-scoped context. Apollo incorporates
  the uploaded document content alongside retrieved database chunks in the
  prompt, enabling personalised clinical decision support for individual
  patient presentations.

---

## 6. Reproducibility

```
Git Commit SHA: 63ddc5422404
Random Seed: 42
llama-cpp-python: 0.2.85
sentence-transformers: 3.0.1
chromadb: 0.5.3
fastapi: 0.111.0
Python: 3.x (see backend/venv)
```

To reproduce from scratch:
1. `git clone https://github.com/itsebuka/Ebuka-s-adtc-2026`
2. `bash download_model.sh`
3. `cd backend && python -m venv venv && venv\Scripts\pip install -r requirements.txt`
4. `double-click start.bat` (or run `.\start.ps1` in PowerShell)

The `start.ps1` launcher automatically detects a missing ChromaDB and
runs `ingest.py` on first launch. On subsequent launches it skips
ingestion and loads the cached database directly.

---

## 7. Testing Methodology

Apollo was evaluated using a dual-track testing methodology combining
automated backend tests and structured manual evaluations.

**Backend functional testing** was performed by issuing HTTP POST requests
directly to the `/chat` endpoint from the command line and inspecting the
streamed SSE output. This allowed precise measurement of retrieval quality
(via the `X-RAG-Top-Similarity` response header), generation speed (tokens/sec
logged by the backend), and response completeness.

**Structured accuracy evaluation** used a curated set of domain-specific
test questions designed to probe failure modes rather than strengths:
- Questions requiring comparison between two closely-named entities
  (e.g., M1 vs. CM2 proteins in Influenza C — requiring symmetric recall)
- Questions about viral evolutionary constraints requiring structural
  genome-level knowledge (overlapping ORFs, RNA secondary structure)
- Questions from traditional medicine and homeopathy domains
  (e.g., Echinacea purpurea homeopathic indications — 82.67% top similarity)
- Complex multi-part virology questions requiring 500+ token structured answers

**Independent third-party evaluation** was conducted by submitting selected
Apollo responses to Gemini Flash for structured critique (simulating a
peer-review process). This produced actionable feedback including:
- Identification of one-sided comparative reasoning (describing CM2 in detail
  while omitting the Influenza A M1/M2 splicing contrast)
- Identification of generic teleological filler language used in place of
  concrete biochemical mechanisms
- Identification of a structural confusion between M2 and CM2 membrane topology

Each identified failure was directly converted into a new system prompt
guardrail (see Section 2.5), demonstrating a closed-loop engineering
methodology. The third-party evaluator assigned a baseline score of 5.5/10
on the initial Influenza C comparative question; the post-fix score is
expected to exceed 8/10 on the same question with the updated pipeline.
