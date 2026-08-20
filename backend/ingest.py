"""
Apollo Advanced RAG Ingestion Pipeline — Structure-Aware Chunking & Metadata Enrichment
========================================================================================
Author: Built for ADTC 2026 — Team: Eleogu Chukwuebuka Joseph

Features:
1. Structure-Aware Chunking: Splits text at markdown headers, tables, lists, and semantic boundaries.
2. Clinical Metadata Enrichment: Automatically tags chunks with:
   - source_doc
   - source_version
   - age_band (neonatal, young_infant, toddler, older_child, adult, all)
   - condition_substance
   - last_reviewed_date
3. Protocol Ingestion: Ingests config/clinical_protocol.yaml as explicit, structured reference chunks.
4. Versioning & Deduplication: Filters out duplicate/superseded versions of identical files.
"""

import os
import sys

# Configure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure offline operation and single-threaded OpenBLAS memory stability on Windows
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import sqlite3
import chromadb
import yaml
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_DIR = Path(__file__).parent
REPO_ROOT = BACKEND_DIR.parent
KNOWLEDGE_DIR = BACKEND_DIR / "data" / "knowledge"
CONFIG_DIR = REPO_ROOT / "config"
CHROMA_DB_DIR = BACKEND_DIR / "chroma_db"
SQLITE_DB_PATH = BACKEND_DIR / "fts.db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "apollo_medical_knowledge"

PARENT_CHUNK_SIZE = 1200
PARENT_CHUNK_OVERLAP = 250
CHILD_CHUNK_SIZE = 512
CHILD_CHUNK_OVERLAP = 128


# ─────────────────────────────────────────────────────────────────────────────
# METADATA EXTRACTION & NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def clean_document_title(filename: str) -> str:
    """Strips artifact IDs and numeric suffixes from filenames."""
    base = re.sub(r"\.(pdf|txt|yaml|yml)$", "", filename, flags=re.IGNORECASE).strip()
    base = re.sub(r"\s*\(\d+\)$", "", base)  # Remove " (1)" duplicate suffixes
    base = re.sub(r"[-_]\d{3,5}$", "", base)
    base = base.replace("-", " ").replace("_", " ")
    
    parts = base.split()
    cleaned_words = []
    for p in parts:
        if re.fullmatch(r"\d{4}", p):
            year = int(p)
            if year < 1900 or year > 2030:
                continue
        cleaned_words.append(p.capitalize())
        
    title_str = " ".join(cleaned_words)
    return title_str if title_str else filename


def infer_age_band(text: str) -> str:
    """Determines applicable age band for a clinical text chunk."""
    t_low = text.lower()
    if any(k in t_low for k in ["neonate", "neonatal", "newborn", "<2 months", "< 2 months", "birth to 2 months"]):
        return "neonatal"
    if any(k in t_low for k in ["young infant", "2-11 months", "2 to 11 months", "infant"]):
        return "young_infant"
    if any(k in t_low for k in ["toddler", "1-5 years", "1 to 5 years", "12-59 months"]):
        return "toddler"
    if any(k in t_low for k in ["child", "pediatric", "paediatric", "5-12 years"]):
        return "older_child"
    if any(k in t_low for k in ["adult", "postpartum", "pregnancy", "trimester", "maternal", "geriatric"]):
        return "adult"
    return "all"


def infer_condition_substance(text: str) -> str:
    """Classifies chunk into a primary condition or substance protocol."""
    t_low = text.lower()
    if "button battery" in t_low or "battery" in t_low:
        return "button_battery"
    if "paracetamol" in t_low or "acetaminophen" in t_low:
        return "paracetamol_overdose"
    if "chest indrawing" in t_low or "respiratory distress" in t_low or "fast breathing" in t_low:
        return "respiratory_distress"
    if "diarrhea" in t_low or "diarrhoea" in t_low or "dehydration" in t_low:
        return "gastroenteritis"
    if "pre-eclampsia" in t_low or "eclampsia" in t_low:
        return "pre_eclampsia"
    if "stroke" in t_low or "facial droop" in t_low:
        return "stroke"
    return "general"


def extract_section_header(text: str) -> str:
    lines = text.splitlines()
    for line in lines:
        line_s = line.strip()
        if line_s.startswith("#") or line_s.lower().startswith("chapter") or line_s.lower().startswith("section"):
            return re.sub(r"^#+\s*", "", line_s)
    return "General Section"


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURE-AWARE CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Splits text into chunks respecting markdown, tables, and list structure."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            search_from = start + int(chunk_size * 0.65)
            for break_char in ['\n## ', '\n### ', '\n---', '\n\n', '\n|', '\n- ', '\n* ', '. ', '! ', '? ']:
                break_pos = text.rfind(break_char, search_from, end)
                if break_pos != -1:
                    end = break_pos + len(break_char)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break
        start = end - overlap

    return chunks


def build_parent_child_hierarchy(
    filename: str, text: str, page_num: int, domain: str = "GENERAL",
    source_doc: str | None = None, source_version: str = "1.0.0",
    age_band_override: str | None = None, condition_override: str | None = None,
    last_reviewed_date: str = "2026-08-20",
) -> List[Dict[str, Any]]:
    """Creates Parent chunks, then splits into Child chunks with rich metadata."""
    documents_to_insert = []
    doc_title = clean_document_title(filename) if not source_doc else source_doc
    sec_header = extract_section_header(text)
    norm_domain = domain.upper().strip() if domain else "GENERAL"

    parent_chunks = split_text(text, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)

    for p_idx, parent_text in enumerate(parent_chunks):
        parent_id = hashlib.md5(f"{filename}_{page_num}_{p_idx}_{parent_text}".encode("utf-8")).hexdigest()
        chunk_age_band = age_band_override or infer_age_band(parent_text)
        chunk_condition = condition_override or infer_condition_substance(parent_text)

        child_chunks = split_text(parent_text, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)

        for c_idx, child_text in enumerate(child_chunks):
            child_id = f"{parent_id}_child_{c_idx}"
            metadata = {
                "source_file": filename,
                "document_title": doc_title,
                "section_header": sec_header,
                "page_number": page_num,
                "parent_id": parent_id,
                "parent_text": parent_text,
                "domain": norm_domain,
                "source_doc": doc_title,
                "source_version": source_version,
                "age_band": chunk_age_band,
                "condition_substance": chunk_condition,
                "last_reviewed_date": last_reviewed_date,
            }
            documents_to_insert.append({
                "id": child_id,
                "text": child_text,
                "metadata": metadata
            })

    return documents_to_insert


def load_protocol_chunks() -> List[Dict[str, Any]]:
    """Loads and chunks clinical_protocol.yaml directly into structured reference chunks."""
    protocol_path = CONFIG_DIR / "clinical_protocol.yaml"
    if not protocol_path.exists():
        return []

    with open(protocol_path, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)

    version = str(proto.get("version", "1.0.0"))
    rev_date = str(proto.get("last_reviewed_date", "2026-08-20"))
    chunks = []

    # 1. Respiratory rate thresholds by age band
    for band in proto.get("respiratory_rate_thresholds", []):
        band_name = band["band"]
        lbl = band["label"]
        thresh = band["fast_breathing_threshold"]
        text = f"CLINICAL PROTOCOL: Fast Breathing Threshold for {lbl} is >={thresh} breaths/min. Severe danger sign: chest indrawing with fast breathing."
        chunks.extend(build_parent_child_hierarchy(
            filename="clinical_protocol.yaml",
            text=text,
            page_num=1,
            domain="PEDIATRICS",
            source_doc="clinical_protocol.yaml",
            source_version=version,
            age_band_override=band_name,
            condition_override="respiratory_distress",
            last_reviewed_date=rev_date,
        ))

    # 2. Substance protocols
    substances = proto.get("substance_protocols", {})
    if "button_battery" in substances:
        bb = substances["button_battery"]
        hp = bb.get("honey_protocol", {})
        bb_text = (
            f"CLINICAL PROTOCOL: Button Battery Ingestion. Emergency: {bb.get('emergency')}. "
            f"Pre-hospital honey protocol: {hp.get('dose_ml')}mL every {hp.get('frequency_minutes')}min "
            f"(max {hp.get('max_doses')} doses) ONLY if age>={hp.get('eligible_min_age_months')}mo "
            f"and ingestion<{hp.get('eligible_max_hours_since_ingestion')}h. Warning: {hp.get('warning')}. "
            f"Strict NPO (nothing by mouth) for infants under 12 months."
        )
        chunks.extend(build_parent_child_hierarchy(
            filename="clinical_protocol.yaml",
            text=bb_text,
            page_num=2,
            domain="TOXICOLOGY",
            source_doc="clinical_protocol.yaml",
            source_version=version,
            age_band_override="all",
            condition_override="button_battery",
            last_reviewed_date=rev_date,
        ))

    if "paracetamol_overdose" in substances:
        para = substances["paracetamol_overdose"]
        p_text = f"CLINICAL PROTOCOL: Paracetamol Overdose. Emergency: {para.get('emergency')}. Antidote: {para.get('antidote')} (most effective within {para.get('time_window_hours')} hours of ingestion)."
        chunks.extend(build_parent_child_hierarchy(
            filename="clinical_protocol.yaml",
            text=p_text,
            page_num=3,
            domain="TOXICOLOGY",
            source_doc="clinical_protocol.yaml",
            source_version=version,
            age_band_override="all",
            condition_override="paracetamol_overdose",
            last_reviewed_date=rev_date,
        ))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# INGESTION & DATABASE SYNC
# ─────────────────────────────────────────────────────────────────────────────

def load_and_chunk_files(knowledge_dir: Path) -> List[Dict[str, Any]]:
    """Reads PDFs and TXT files recursively, excluding duplicate copies."""
    all_chunks = []
    
    # First, ingest the authoritative clinical_protocol.yaml
    protocol_chunks = load_protocol_chunks()
    all_chunks.extend(protocol_chunks)
    print(f"  [PROTOCOL] Loaded {len(protocol_chunks)} structured reference chunks from clinical_protocol.yaml")

    txt_files = list(knowledge_dir.rglob("*.txt"))
    pdf_files = list(knowledge_dir.rglob("*.pdf"))
    raw_files = sorted(txt_files + pdf_files)

    # Filter duplicate file versions (e.g. "Betts... (1).pdf")
    seen_bases = set()
    filtered_files = []
    for f in raw_files:
        if f.name.startswith(".~lock") or f.name.startswith("~$") or f.name.endswith("#"):
            continue
        base_name = re.sub(r"\s*\(\d+\)", "", f.stem).strip()
        if base_name in seen_bases:
            print(f"  [DEDUP] Skipping duplicate file copy: {f.name}")
            continue
        seen_bases.add(base_name)
        filtered_files.append(f)

def process_and_index_file(
    file_path: Path,
    knowledge_dir: Path,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    existing_ids: set,
    conn: sqlite3.Connection,
) -> int:
    """Chunks, embeds, and indexes a single document file with real-time feedback."""
    rel_path = file_path.relative_to(knowledge_dir)
    domain = rel_path.parent.name if rel_path.parent.name else "general"
    chunks: List[Dict[str, Any]] = []

    if file_path.suffix.lower() == '.pdf':
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                except Exception:
                    page_text = None
                if page_text:
                    page_chunks = build_parent_child_hierarchy(file_path.name, page_text, page_num=i+1, domain=domain)
                    chunks.extend(page_chunks)
        except Exception as fe:
            print(f"    ⚠ Could not open PDF {file_path.name}: {fe}", flush=True)
            return 0
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        section_size = 3000
        sections = [content[i:i + section_size].strip() for i in range(0, len(content), section_size)]
        for pseudo_page, section_text in enumerate(sections, start=1):
            if not section_text:
                continue
            section_chunks = build_parent_child_hierarchy(file_path.name, section_text, page_num=pseudo_page, domain=domain)
            chunks.extend(section_chunks)

    if not chunks:
        return 0

    # Filter out already existing chunks
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]
    if not new_chunks:
        print(f"    ✓ All {len(chunks)} chunks already up to date.", flush=True)
        return 0

    # Embed new chunks in batches
    texts_to_embed = [c["text"] for c in new_chunks]
    new_embeddings = embedding_model.encode(
        texts_to_embed,
        batch_size=256,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    for i, chunk in enumerate(new_chunks):
        chunk["embedding"] = new_embeddings[i]
        existing_ids.add(chunk["id"])

    # Commit to ChromaDB
    batch_size = 2000
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i+batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    # Commit to SQLite FTS5
    c = conn.cursor()
    c.executemany('''
        INSERT INTO chunks (id, text, parent_id, parent_text, source_file, page_number, domain, age_band, condition_substance, source_doc, source_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', [(
        chk["id"],
        chk["text"],
        chk["metadata"]["parent_id"],
        chk["metadata"]["parent_text"],
        chk["metadata"]["source_file"],
        str(chk["metadata"]["page_number"]),
        chk["metadata"].get("domain", "GENERAL"),
        chk["metadata"].get("age_band", "all"),
        chk["metadata"].get("condition_substance", "general"),
        chk["metadata"].get("source_doc", chk["metadata"]["source_file"]),
        chk["metadata"].get("source_version", "1.0.0"),
    ) for chk in new_chunks])
    conn.commit()

    return len(new_chunks)


def main():
    print("=" * 65)
    print("  APOLLO — Advanced RAG Pipeline (Progressive Ingestion & Sync)")
    print("=" * 65)

    reset_db = "--reset" in sys.argv

    print("\n[INIT] Loading local SentenceTransformer embeddings...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("  ✓ SentenceTransformer loaded.")

    print("\n[INIT] Connecting to ChromaDB & SQLite FTS5...")
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    if reset_db:
        print("  [RESET] --reset detected: Re-creating collections...")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        c = conn.cursor()
        if reset_db:
            c.execute("DROP TABLE IF EXISTS chunks")
            conn.commit()

        c.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                id UNINDEXED,
                text,
                parent_id UNINDEXED,
                parent_text UNINDEXED,
                source_file UNINDEXED,
                page_number UNINDEXED,
                domain UNINDEXED,
                age_band UNINDEXED,
                condition_substance UNINDEXED,
                source_doc UNINDEXED,
                source_version UNINDEXED
            )
        ''')
        conn.commit()

        # Load existing IDs to avoid re-embedding
        existing_ids = set()
        if collection.count() > 0:
            existing_ids = set(collection.get(include=[])["ids"])

        # 1. Ingest Authoritative Clinical Protocol Chunks
        print("\n[STEP 1/2] Indexing clinical_protocol.yaml reference chunks...", flush=True)
        proto_chunks = load_protocol_chunks()
        new_proto_map = {chk["id"]: chk for chk in proto_chunks if chk["id"] not in existing_ids}
        new_proto = list(new_proto_map.values())
        if new_proto:
            texts = [c["text"] for c in new_proto]
            embeddings = embedding_model.encode(texts, normalize_embeddings=True).tolist()
            collection.add(
                ids=[c["id"] for c in new_proto],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c["metadata"] for c in new_proto],
            )
            c.executemany('''
                INSERT INTO chunks (id, text, parent_id, parent_text, source_file, page_number, domain, age_band, condition_substance, source_doc, source_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [(
                chk["id"], chk["text"], chk["metadata"]["parent_id"], chk["metadata"]["parent_text"],
                chk["metadata"]["source_file"], str(chk["metadata"]["page_number"]),
                chk["metadata"].get("domain", "GENERAL"), chk["metadata"].get("age_band", "all"),
                chk["metadata"].get("condition_substance", "general"),
                chk["metadata"].get("source_doc", "clinical_protocol.yaml"),
                chk["metadata"].get("source_version", "1.0.0"),
            ) for chk in new_proto])
            conn.commit()
            for chk in new_proto:
                existing_ids.add(chk["id"])
            print(f"  ✓ {len(new_proto)} clinical protocol reference chunks indexed.")
        else:
            print(f"  ✓ Protocol chunks already up to date ({len(proto_chunks)} chunks).")

        # 2. Ingest Document Files
        txt_files = list(KNOWLEDGE_DIR.rglob("*.txt"))
        pdf_files = list(KNOWLEDGE_DIR.rglob("*.pdf"))
        raw_files = sorted(txt_files + pdf_files)

        seen_bases = set()
        filtered_files = []
        for f in raw_files:
            if f.name.startswith(".~lock") or f.name.startswith("~$") or f.name.endswith("#"):
                continue
            base_name = re.sub(r"\s*\(\d+\)", "", f.stem).strip()
            if base_name in seen_bases:
                continue
            seen_bases.add(base_name)
            filtered_files.append(f)

        print(f"\n[STEP 2/2] Ingesting {len(filtered_files)} documents from knowledge repository...", flush=True)
        total_inserted = 0
        t0 = time.time()

        for idx, file_path in enumerate(filtered_files, start=1):
            t_file = time.time()
            rel_path = file_path.relative_to(KNOWLEDGE_DIR)
            domain = rel_path.parent.name if rel_path.parent.name else "general"
            print(f"[{idx:02d}/{len(filtered_files):02d}] {file_path.name[:50]:<50} [{domain}]", end=" ... ", flush=True)
            
            inserted = process_and_index_file(
                file_path=file_path,
                knowledge_dir=KNOWLEDGE_DIR,
                collection=collection,
                embedding_model=embedding_model,
                existing_ids=existing_ids,
                conn=conn,
            )
            total_inserted += inserted
            el = time.time() - t_file
            print(f"+{inserted} chunks ({el:.1f}s)", flush=True)

        total_time = time.time() - t0
        print("\n" + "=" * 65)
        print("  INGESTION & SYNC COMPLETE")
        print("=" * 65)
        print(f"  Total Files Processed: {len(filtered_files)}")
        print(f"  Newly Indexed Chunks : {total_inserted}")
        print(f"  Total Index Size     : {collection.count()} vectors")
        print(f"  Elapsed Time         : {total_time:.1f}s")
        print("=" * 65)


if __name__ == "__main__":
    main()
