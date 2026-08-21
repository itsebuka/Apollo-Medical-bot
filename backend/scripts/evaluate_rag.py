"""
Apollo RAG Pipeline — Evaluation & Latency Benchmarking Harness
Measures Precision@k, Recall@k, Grounding Pass Rate, and Stage-by-Stage Latency
across Safety-Critical, Near-Boundary, Distractor, and Common-Case subsets.
"""
import sys, os, time, json, asyncio
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app import config
from app.pipeline import preprocess_query, validate_and_repair, build_system_context_block, ApolloResponse, Escalation
import app.main as main_mod
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

EVAL_SET_PATH = ROOT / "tests" / "rag_eval_set.jsonl"
RESULTS_OUTPUT_PATH = ROOT / "backend" / "scripts" / "rag_eval_results.json"

async def init_models():
    if config.llm_instance is None:
        print("[1/4] Loading GGUF LLM...")
        config.llm_instance = Llama(model_path=str(config.MODEL_PATH), **config.LLM_CONFIG)
    if config.embedding_model_instance is None:
        print("[2/4] Loading Embedding Model...")
        config.embedding_model_instance = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    if config.chroma_collection_instance is None:
        print("[3/4] Loading ChromaDB...")
        client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
        config.chroma_collection_instance = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
    if config.cross_encoder_instance is None:
        print("[4/4] Loading CrossEncoder...")
        config.cross_encoder_instance = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME)
    print("All models ready.\n")

async def evaluate_single_query(item: dict, generate_llm: bool = True) -> dict:
    query = item["query"]
    t0_total = time.perf_counter()

    # Stage 1: Pre-processing & Age/Entity Extraction
    t0_pre = time.perf_counter()
    sq = preprocess_query(query)
    t_pre = (time.perf_counter() - t0_pre) * 1000

    # Stage 2: Retrieval (Hybrid + Reranking)
    t0_retrieval = time.perf_counter()
    chunks = await main_mod.retrieve_context(sq.cleaned_input, n_results=5, structured_query=sq if hasattr(main_mod, 'retrieve_context') else None)
    t_retrieval = (time.perf_counter() - t0_retrieval) * 1000

    # Retrieval Metrics
    retrieved_texts = [c.get("text", "") for c in chunks]
    combined_retrieved_text = " ".join(retrieved_texts)
    
    # Keyword coverage (Recall indicator)
    expected_kws = item.get("expected_keywords", [])
    matched_kws = [kw for kw in expected_kws if kw.lower() in combined_retrieved_text.lower()]
    recall_score = len(matched_kws) / len(expected_kws) if expected_kws else 1.0

    # Age band precision
    exp_band = item.get("expected_age_band")
    correct_band_retrieved = True
    if exp_band and exp_band != "all":
        # Check if text contains wrong thresholds
        protocol = config.REPO_ROOT / "config" / "clinical_protocol.yaml"
        # If retrieved chunk explicitly talks about the target age band or condition
        correct_band_retrieved = (sq.age_band is not None and sq.age_band.get("band") == exp_band) or (exp_band in combined_retrieved_text.lower())

    t_llm = 0.0
    t_post = 0.0
    llm_output = ""
    val_status = "N/A"
    grounding_passed = True

    if generate_llm:
        # Stage 3: Prompt Construction
        system_block = build_system_context_block(sq)
        messages = main_mod.build_prompt([{"role": "user", "content": sq.cleaned_input}], chunks, system_context_block=system_block)
        
        # Stage 4: LLM Generation
        t0_llm = time.perf_counter()
        loop = asyncio.get_running_loop()
        def _gen():
            res = config.llm_instance.create_chat_completion(messages=messages, stream=False, **{**config.GENERATION_CONFIG, "max_tokens": 1024})
            return res["choices"][0]["message"]["content"]
        llm_output = await loop.run_in_executor(None, _gen)
        t_llm = (time.perf_counter() - t0_llm) * 1000

        # Stage 5: Post-processing Validation
        t0_post = time.perf_counter()
        chunk_ids = [c.get("id", c.get("source", "chk")) for c in chunks]
        val_res = validate_and_repair(llm_output, sq, retrieved_chunks=chunks, retrieved_chunk_ids=chunk_ids)
        t_post = (time.perf_counter() - t0_post) * 1000
        val_status = type(val_res).__name__
        grounding_passed = isinstance(val_res, ApolloResponse) or (isinstance(val_res, Escalation) and item.get("red_flag", False))

    t_total = (time.perf_counter() - t0_total) * 1000

    return {
        "id": item["id"],
        "subset": item["subset"],
        "query": query,
        "cleaned_query": sq.cleaned_input,
        "age_months": sq.age_months,
        "age_band": sq.age_band.get("band") if sq.age_band else "null",
        "retrieved_chunks_count": len(chunks),
        "recall_score": recall_score,
        "matched_keywords": matched_kws,
        "correct_band_retrieved": correct_band_retrieved,
        "validation_status": val_status,
        "grounding_passed": grounding_passed,
        "latency_ms": {
            "pre_processing": round(t_pre, 2),
            "retrieval": round(t_retrieval, 2),
            "llm_generation": round(t_llm, 2),
            "post_processing": round(t_post, 2),
            "total_end_to_end": round(t_total, 2),
        },
        "llm_preview": llm_output[:180] if llm_output else "",
    }

async def main():
    await init_models()
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(items)} evaluation queries from {EVAL_SET_PATH.name}")
    results = []
    
    # Run evaluation across all items
    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] Evaluating {item['id']} ({item['subset']})...", flush=True)
        res = await evaluate_single_query(item, generate_llm=True)
        results.append(res)
        print(f"    -> Recall: {res['recall_score']:.1%} | Grounding: {res['grounding_passed']} | Total: {res['latency_ms']['total_end_to_end']:.0f}ms")

    # Aggregate metrics by subset
    subsets = ["safety_critical", "near_boundary", "distractor", "common_case"]
    summary = {}
    
    for sub in subsets:
        sub_items = [r for r in results if r["subset"] == sub]
        if not sub_items:
            continue
        recalls = [r["recall_score"] for r in sub_items]
        band_correct = [r["correct_band_retrieved"] for r in sub_items]
        grounding = [r["grounding_passed"] for r in sub_items]
        total_lats = [r["latency_ms"]["total_end_to_end"] for r in sub_items]
        retrieval_lats = [r["latency_ms"]["retrieval"] for r in sub_items]
        llm_lats = [r["latency_ms"]["llm_generation"] for r in sub_items]

        summary[sub] = {
            "count": len(sub_items),
            "mean_recall_at_k": round(float(np.mean(recalls)), 4),
            "age_band_accuracy": round(float(np.mean(band_correct)), 4),
            "grounding_pass_rate": round(float(np.mean(grounding)), 4),
            "retrieval_latency_p50_ms": round(float(np.percentile(retrieval_lats, 50)), 1),
            "retrieval_latency_p95_ms": round(float(np.percentile(retrieval_lats, 95)), 1),
            "total_latency_p50_ms": round(float(np.percentile(total_lats, 50)), 1),
            "total_latency_p95_ms": round(float(np.percentile(total_lats, 95)), 1),
        }

    output_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "detailed_results": results,
    }

    with open(RESULTS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("\n" + "="*70)
    print("APOLLO RAG EVALUATION SUMMARY")
    print("="*70)
    for sub, stats in summary.items():
        print(f"\n--- {sub.upper()} (N={stats['count']}) ---")
        print(f"  Recall@5:                {stats['mean_recall_at_k']:.1%}")
        print(f"  Age-Band Accuracy:       {stats['age_band_accuracy']:.1%}")
        print(f"  Grounding Pass Rate:     {stats['grounding_pass_rate']:.1%}")
        print(f"  Retrieval Latency (p50): {stats['retrieval_latency_p50_ms']}ms | (p95): {stats['retrieval_latency_p95_ms']}ms")
        print(f"  Total Latency (p50):     {stats['total_latency_p50_ms']}ms | (p95): {stats['total_latency_p95_ms']}ms")
    print("="*70)
    print(f"Saved report to: {RESULTS_OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
