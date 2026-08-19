# Technical Report — Apollo: An Offline-First Clinical Decision Support System for Nigerian Healthcare Workers

**Team ID:** ADTC-2026-APOLLO
**Submitter:** Eleogu Chukwuebuka Joseph
**Domain:** healthcare_medical
**Model:** Meta-Llama-3.1-8B-Instruct-Q4_K_M
**GitHub:** https://github.com/itsebuka/Apollo-Medical-bot

---

## 1. Problem Statement & Operational Scope

Nigeria's healthcare system faces a structural challenge that is both systemic and immediate: the severe maldistribution of specialist medical knowledge relative to where patients actually present. According to the World Health Organization, Nigeria operates with approximately 4 doctors per 10,000 people — far below the WHO-recommended minimum of 10. The burden falls disproportionately on nurses, community health extension workers (CHEWs), and junior medical officers who must make consequential triage decisions, often without access to specialist consultants, reliable reference texts, or the internet.

The connectivity problem compounds this. Large portions of Nigeria's healthcare delivery occurs in primary and secondary facilities located in peri-urban and rural areas where mobile broadband is intermittent or prohibitively expensive. Even in urban teaching hospitals, electricity supply is unreliable, and internet access is frequently disrupted. Cloud-hosted clinical AI tools — including those powered by GPT-4, Claude, or Gemini — are therefore unusable in precisely the environments where AI assistance is most needed.

Apollo directly addresses this gap. It is a Retrieval-Augmented Generation (RAG) clinical decision support & research engine designed to run entirely offline on a consumer-grade Windows laptop. It gives a healthcare worker in a rural primary health care centre the same quality of structured clinical reference access that a doctor at a Lagos teaching hospital might have access to via the internet — with zero dependency on network connectivity during operation.

**Target User:** Nigerian healthcare workers — specifically nurses, community health officers, resident doctors, and junior clinicians — who need rapid, structured access to clinical reference information across infectious disease, pharmacology, pathophysiology, and related domains without internet access.

**Scope of Clinical Coverage:** Apollo's knowledge base spans 18 medical domains, with particular emphasis on conditions that carry high prevalence or clinical urgency in the Nigerian epidemiological context, including HIV/AIDS, tuberculosis, malaria comorbidities, internal medicine, physical examination, emergency medicine, pediatrics, diabetes, hypertension, schizophrenia, dementia, ADHD, depression, anxiety, asthma, parasitic diseases, and multiple forms of cancer. The system's Traditional Medicine and Homeopathy domains directly reflect the reality that a meaningful proportion of Nigerian patients arrive having already used traditional remedies, and clinicians need accurate pharmacodynamic context to assess for interactions.

**Language Scope:** The system is designed to accept queries in English, Nigerian Pidgin (Naija), Hausa, Yoruba, and Igbo — the four dominant language groups across Nigeria's geopolitical zones — making it accessible to a wider range of healthcare workers regardless of their level of formal English fluency.

---

## 2. System Architecture & Technical Design Decisions

### 2.1 Model Selection & Quantization Strategy

Three models were evaluated before settling on the final architecture:

- **Candidate 1: Meditron-7B** (Llama 2 medical fine-tune). Rejected on output quality grounds; produced unstructured outputs and struggled with multi-hop clinical reasoning.
- **Candidate 2: Gemma 7B** (Google). Rejected on output quality grounds; multi-rule instruction following was weaker than Llama 3.1 at the same parameter count.
- **Final Selection: Meta-Llama-3.1-8B-Instruct (Q4_K_M)**. Chosen for superior instruction following (handling complex multi-rule behavioral guardrails), broad base knowledge, and memory budget compatibility (~4.7 GB GGUF weight size, leaving ample RAM headroom).

**Quantization Level (Q4_K_M):** 4-bit quantization with K-quant mixed precision was selected as the optimal point on the quality-vs-memory Pareto frontier. Peak RSS measures ~6.17 GB total (model + embeddings + re-ranker + OS overhead), leaving ~1.83 GB of safety margin within the 8 GB evaluation limit.

---

### 2.2 Advanced RAG Pipeline Architecture

Apollo implements a multi-stage, high-precision retrieval pipeline engineered to eliminate factual hallucinations, broken cross-chunk logic, missing biochemical mechanisms, and reference fabrication:

#### Stage 1 — Ingestion: Sliding Window Parent-Child Chunking with Semantic Boundaries (`ingest.py`)
- **Parent Chunks (1,000 characters/tokens, 250 overlap):** Generous sliding window context preserves tabular data, gene/protein maps, and step-by-step enzymatic cascades within contiguous chunks.
- **Child Chunks (256 characters/tokens, 100 overlap):** Precision semantic targets for vector embedding. Dense vectors computed from child chunks drive exact retrieval, while the full parent chunk is attached to metadata and returned to the LLM context window.
- **Semantic Boundary Hierarchy:** Splitting prioritizes Markdown headers (`##`, `###`), section dividers (`---`), double linebreaks (`\n\n`), and bullet points (`- `, `* `) before falling back to sentence breaks (`. `, `! `).
- **Metadata Enrichment & Filename Normalization:** Raw internal filenames (e.g. `molecular-virology-moses-p-adoga-2347.pdf`) are normalized into publication-grade document titles (`Molecular Virology (Moses P. Adoga)`). Each chunk is enriched with `domain` (normalized uppercase tag e.g. `"VIROLOGY"`, `"ANATOMY"`), `document_title`, `section_header`, and `page_number`.
- **Automatic Library Sync & Orphan Cleanup:** `ingest.py` includes `sync_deleted_files()` and `--reset` flag support to automatically purge vector and FTS5 entries for books deleted from disk, keeping the database 100% synchronized with the physical library.

#### Stage 2 — Two-Stage Hybrid Search: Dense Vectors + BM25 Sparse Keywords
- **Dense Vector Search (ChromaDB):** Cosine similarity over 384-dimensional embeddings generated by `sentence-transformers/all-MiniLM-L6-v2`.
- **Sparse Keyword Search (SQLite FTS5):** BM25 full-text keyword ranking over an indexed FTS5 table, ensuring exact matches for proper nouns, gene names, and biochemical terms (`peritrophic`, `metalloproteinase`, `HEXXH`, `mucin`, `cricothyroid`).
- **Reciprocal Rank Fusion (RRF):** Candidate results from dense and sparse retrievers are merged using RRF ($k=60$).

#### Stage 3 — Query Intent Routing & Hybrid Term Expansion
- **Biochemical Mechanism Router:** Queries seeking "biochemical mechanism", "mode of action", "pathogenicity", or "cleavage" automatically expand with high-yield enzymatic terms (`mechanism`, `enzymatic`, `cleavage`, `substrate`, `pathway`, `catalytic`, `metalloproteinase`, `zinc`, `endopeptidase`, `mucin`, `peritrophic`).
- **Preamble Sanitization:** `_sanitize_clinical_query()` automatically strips preamble headers/noise (e.g. `"second attemp at asking..."`) prior to vector search.
- **Domain Scoping:** Pre-filters or score-boosts candidates matching target domain metadata (`domain: "ANATOMY"`, `"PHARMACOLOGY"`, `"PATHOPHYSIOLOGY"`).

#### Stage 4 — Top-$k$ Candidate Expansion ($initial\_k = 12$) & Cross-Encoder Re-Ranking
- Initial candidate retrieval expands the pool to $initial\_k = 12$ candidates.
- Passes candidate chunks through Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) for full joint-attention pair scoring against the original prompt.
- Injects the top 5 to 6 highest-scoring re-ranked chunks into the LLM context window ($n\_results = 5$).

#### Stage 5 — Dynamic Token Allocation ($max\_tokens = 2560$)
- Complex queries (mechanism, comparison, pathway, structural questions) receive a dedicated **2560 max tokens** budget, guaranteeing that reference lists and multi-step clinical cascades are never truncated mid-answer.

---

### 2.3 Comprehensive Knowledge Base

Apollo's offline knowledge base contains **66 major medical reference books, clinical guidelines, and manuals** spanning 18 specialty domains:

- **Core Clinical References:** *Harrison's Principles of Internal Medicine (20th Ed)*, *Bates' Guide to Physical Examination and History Taking*, *Cecil Essentials of Medicine (10th Ed)*, *Nelson Textbook of Pediatrics (21st Ed)*, *The Merck Manual of Diagnosis and Therapy (19th Ed)*, *Oxford Handbook of Emergency Medicine (4th Ed)*, MSF *Clinical Guidelines*, MSF *Essential Drugs*, WHO *Surgical Care at the District Hospital*, HHS *Guide for HIV/AIDS Clinical Care*, FHI 360 *HIV/AIDS Care and Treatment*.
- **Anatomy & Physiology:** *Anatomy and Physiology 2e* (OpenStax), *Anatomy & Physiology* (Biga et al.), *Fundamentals of Anatomy and Physiology* (Chrucik et al.).
- **Pharmacology & Nursing:** *OpenStax Pharmacology for Nurses*, *Nursing Fundamentals 2e* (WTCS), *Pharmacology Review* (DAMS), *General Pharmacology* (Germanyuk et al.).
- **Microbiology, Virology & Bacteriology:** *OpenStax Microbiology* (Parker et al.), *Microbiology for Allied Health Students* (Smith & Selby), *Molecular Virology* (Adoga), *Notes on Medical Virology* (Aldigs).
- **Parasitology & Pathophysiology:** *Despommier's Parasitic Diseases* (Griffin et al.), *Concepts in Animal Parasitology* (Gardner et al.), *Intestinal Parasites* (Dogan), *General Pathophysiology*.
- **Homeopathy & Traditional Medicine:** Boericke's *Pocket Manual of Homeopathic Materia Medica*, Hering's *Homeopathic Domestic Physician*, *A Guide to Medicinal Plants of Appalachia* (Krochmal et al.), *Herbal Medicine* (Builders et al.), *Natural Medicinal Plants* (El-Shemy).
- **Genetics & Neuroscience:** *Human Genetics* (Nikitin), *Introduction to Genetics* (Ramroop Singh), *Chromosomes, Genes and Traits* (Simons), *Introduction to Epigenetics*, *Introduction to Neuroscience* (Hedges).

---

### 2.4 System Prompt & Strict Behavioral Guardrails (`config.py`)

Apollo's generation prompt enforces **8 strict operational directives**:

1. **Strict Factual Grounding (Zero Tolerance for Hallucination):** Answers strictly using explicit facts from context chunks; forbids extrapolating or substituting general pre-training memory for missing numerical lengths or mechanisms.
2. **Comparative Questions (Dual Baseline Rule):** When contrasting Entity A vs. Entity B, explicitly states the genomic/structural baseline of BOTH entities before drawing conclusions.
3. **Entity & Nomenclature Discrimination:** Differentiates closely named variants (e.g., M1 [242 aa] vs. M1' [259 aa] vs. P42/P44 [374 aa]).
4. **Publication-Grade Citation Integrity:** Every claim includes inline bracket citations matching clean document metadata: `[Source: <Clean Document Title> (Page <N>)]`. Prohibits interpreting numeric database IDs in filenames as publication years.
5. **Biochemical & Enzymatic Mechanism Protocol:** When asked for a disease/pathogenicity mechanism, explicitly identifies (a) specific enzyme class (e.g., metalloproteinase), (b) target substrate (e.g., peritrophic membrane mucin), and (c) structural alteration. Forbids hand-waving with generic terms like "genome collinearity" or "evolutionary pressure".
6. **Output Structure & Token Management:** Clean section headers, bulleted lists, and complete reference lists without mid-sentence cutoffs.
7. **Zero Fabrication Rule:** Never invents objective clinical findings (vitals, lab results).
8. **Viral Evolutionary Constraint Protocol:** Probes context for structural genome-level limitations (overlapping ORFs, RNA secondary structures, host DNA repair rcDNA-to-cccDNA conversion) rather than generic error rates.

---

## 3. Frontend UX & Accessibility Enhancements

Apollo's React frontend (`frontend/src/`) was redesigned to deliver a premium, modern clinical interface:

- **Cobalt Blue & Electric Azure Palette (`#2563EB` / `#38BDF8`):** Upgraded theme and snake SVG icon (`apollo-icon.svg`) from legacy green to modern high-contrast blue tones.
- **High-Contrast Selection Styling:** Updated selection contrast (`selection:bg-slate-950 dark:selection:text-sky-200`) so text never disappears when highlighted by clinicians.
- **One-Click User Prompt Copying:** Integrated interactive copy button on user message bubbles in `MessageBubble.jsx` with instant clipboard write and checkmark feedback animation.
- **EHR Summary Export & Text-to-Speech:** One-click copying formatted as clinical handoff notes (`Apollo Triage Summary / Generated: [HH:MM]`), plus zero-RAM browser-native TTS voice synthesis.
- **Real-Time Token Streaming:** Server-Sent Events (SSE) stream responses token-by-token for immediate readability.

---

## 4. Hardware Constraints & Offline Benchmarks

- **Development Hardware:** Intel Core i7-11850H CPU (8 cores), 15.7 GB RAM, Windows 11.
- **Evaluation Envelope:** 4 vCPU, 8 GB RAM, CPU-only inference (`n_gpu_layers: 0`).
- **Peak RAM (RSS):** ~6,170 MB (~1,830 MB RAM safety margin under 8 GB ceiling).
- **Offline Enforcement:** `HF_HUB_OFFLINE=1` set in environment; zero network calls during runtime.

---

## 5. Verification & Reproducibility

```powershell
Git Branch: main
Primary Commits: 
  - 55a5184 (fix: wrap PDF page text extraction in try/except to prevent crash)
  - 279e80e (style: add flush=True to ingest.py for real-time terminal streaming)
  - d8da204 (feat: add automatic orphan file cleanup and --reset flag support)
  - 92b42f5 (feat: implement initial_k = 12 pool expansion & publication-grade labels)
  - 6c15ce3 (feat: bump dynamic max_tokens to 2560 in main.py)
  - 52c0255 (feat: increase N_RESULTS to 5 in config.py)
  - 2151454 (feat: overhaul SYSTEM_PROMPT in config.py with Section 3 directives)
  - 69b9a2e (feat: overhaul ingest.py with sliding window chunking & metadata enrichment)
```

To reproduce database build and launch:
1. `git clone https://github.com/itsebuka/Apollo-Medical-bot`
2. `cd backend && .\venv\Scripts\python.exe -u ingest.py --reset`
3. Execute `.\start.ps1` or double-click `start.bat`.

---

## 6. Summary

Apollo bridges the clinical information gap in resource-constrained environments by delivering publication-grade RAG performance, zero-hallucination guardrails, and a modern offline UI — empowering healthcare workers across Nigeria with instant, trustworthy clinical decision support.
