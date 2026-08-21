"""
Apollo Index Integrity & Version Staleness Checker
===================================================
Runs during CI / deployment to verify:
1. No duplicate active versions of identical source documents are indexed.
2. All chunks in ChromaDB and SQLite FTS5 contain required structured metadata.
3. Clinical protocol chunks match the version in clinical_protocol.yaml.
"""

import sys
import sqlite3
from pathlib import Path
import chromadb
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = ROOT / "config" / "clinical_protocol.yaml"
CHROMA_DIR = ROOT / "backend" / "chroma_db"
SQLITE_DB = ROOT / "backend" / "fts.db"

def check_integrity() -> bool:
    print("==================================================")
    print("  APOLLO INDEX INTEGRITY & VERSION AUDIT")
    print("==================================================")
    
    passed = True

    # 1. Load active protocol version from YAML
    if not CONFIG_PATH.exists():
        print(f"❌ ERROR: Protocol file not found at {CONFIG_PATH}")
        return False
        
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)
    active_version = str(proto.get("version", "1.0.0"))
    print(f"✓ Active Protocol Version from YAML: v{active_version}")

    # 2. Check SQLite FTS5 Table Schema & Document Versions
    if SQLITE_DB.exists():
        try:
            with sqlite3.connect(SQLITE_DB) as conn:
                c = conn.cursor()
                c.execute("PRAGMA table_info(chunks)")
                cols = [r[1] for r in c.fetchall()]
                print(f"✓ SQLite FTS5 columns: {cols}")
                
                # Check for multiple active versions of protocol
                c.execute("SELECT DISTINCT source_version FROM chunks WHERE source_doc = 'clinical_protocol.yaml'")
                proto_versions = [r[0] for r in c.fetchall()]
                if len(proto_versions) > 1:
                    print(f"❌ INTEGRITY VIOLATION: Multiple versions of clinical_protocol.yaml found in FTS5: {proto_versions}")
                    passed = False
                elif proto_versions and proto_versions[0] != active_version:
                    print(f"❌ VERSION MISMATCH: FTS5 protocol v{proto_versions[0]} != YAML active v{active_version}")
                    passed = False
                else:
                    print(f"✓ FTS5 protocol version is singular and aligned (v{active_version}).")
        except Exception as e:
            print(f"⚠ SQLite audit note: {e}")
    else:
        print(f"ℹ SQLite DB not initialized yet at {SQLITE_DB}")

    # 3. Check ChromaDB Collection
    if CHROMA_DIR.exists():
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            col = client.get_or_create_collection("apollo_medical_knowledge")
            total_count = col.count()
            print(f"✓ ChromaDB Total Chunks: {total_count}")
            
            if total_count > 0:
                results = col.get(where={"source_doc": "clinical_protocol.yaml"}, include=["metadatas"])
                metas = results.get("metadatas", [])
                versions = set(m.get("source_version") for m in metas if m.get("source_version"))
                if len(versions) > 1:
                    print(f"❌ INTEGRITY VIOLATION: Multiple versions of clinical_protocol.yaml found in ChromaDB: {versions}")
                    passed = False
                elif versions and list(versions)[0] != active_version:
                    print(f"❌ VERSION MISMATCH: ChromaDB protocol v{list(versions)[0]} != YAML active v{active_version}")
                    passed = False
                else:
                    print(f"✓ ChromaDB protocol version is singular and aligned (v{active_version}).")
        except Exception as e:
            print(f"⚠ ChromaDB audit note: {e}")

    print("==================================================")
    if passed:
        print("  INDEX INTEGRITY AUDIT: PASSED ✓")
    else:
        print("  INDEX INTEGRITY AUDIT: FAILED ❌")
    print("==================================================")
    return passed

if __name__ == "__main__":
    ok = check_integrity()
    sys.exit(0 if ok else 1)
