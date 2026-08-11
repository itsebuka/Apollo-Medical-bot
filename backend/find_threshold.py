import sys
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Setup paths
BACKEND_DIR = Path(__file__).parent
CHROMA_DB_DIR = BACKEND_DIR / "chroma_db"
COLLECTION_NAME = "apollo_medical_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    if not CHROMA_DB_DIR.exists():
        print(f"[ERROR] ChromaDB directory not found at: {CHROMA_DB_DIR}")
        print("        Run 'python ingest.py' first to build the vector database.")
        return

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    
    greetings = [
        "Hello",
        "Hi there",
        "How are you today?",
        "Who created you?",
        "What is your name?",
        "Good morning",
        "I am feeling happy",
        "Tell me a joke",
        "Are you a robot?",
        "What's up?"
    ]
    
    medical_queries = [
        "What is malaria?",
        "Treatment for HIV",
        "How to cure a headache",
        "Symptoms of tuberculosis",
        "Pharmacokinetics of amoxicillin",
        "What are capsomers?",
        "Pathophysiology of diabetes",
        "Homeopathic remedy for cough",
        "Antimicrobial resistance in Nigeria",
        "Diagnosis of typhoid fever"
    ]
    
    print("\n--- GREETINGS (NOISE) ---")
    max_greeting_score = 0.0
    for q in greetings:
        emb = model.encode([q], normalize_embeddings=True).tolist()
        res = collection.query(query_embeddings=emb, n_results=1, include=["distances"])
        if res["distances"] and res["distances"][0]:
            dist = res["distances"][0][0]
            score = 1.0 - dist
            max_greeting_score = max(max_greeting_score, score)
            print(f"Query: '{q}' -> Top Score: {score:.4f}")
            
    print("\n--- MEDICAL QUERIES (SIGNAL) ---")
    min_medical_score = 1.0
    for q in medical_queries:
        emb = model.encode([q], normalize_embeddings=True).tolist()
        res = collection.query(query_embeddings=emb, n_results=1, include=["distances"])
        if res["distances"] and res["distances"][0]:
            dist = res["distances"][0][0]
            score = 1.0 - dist
            min_medical_score = min(min_medical_score, score)
            print(f"Query: '{q}' -> Top Score: {score:.4f}")
            
    print("\n--- MATHEMATICAL THRESHOLD CALCULATION ---")
    print(f"Noise Ceiling (Highest Greeting Score): {max_greeting_score:.4f}")
    print(f"Signal Floor (Lowest Medical Score): {min_medical_score:.4f}")
    
    if max_greeting_score >= min_medical_score:
        print("WARNING: Overlap detected! A perfect linear threshold cannot strictly separate them based on top-1 dense retrieval alone.")
        threshold = (max_greeting_score + min_medical_score) / 2.0
    else:
        threshold = (max_greeting_score + min_medical_score) / 2.0
        print(f"PERFECT MIDPOINT THRESHOLD: {threshold:.4f}")

if __name__ == '__main__':
    main()
