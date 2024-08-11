import chromadb
import random

try:
    client = chromadb.PersistentClient(path='chroma_db')
    collection = client.get_collection('apollo_medical_knowledge')
    
    # We don't want to load all 100k chunks, just get a random sample
    data = collection.get(limit=5000)
    
    docs = data['documents']
    metas = data['metadatas']
    
    print("--- POTENTIAL QUESTION SOURCES ---")
    
    # Pick a few random ones that look like they contain good medical knowledge
    # i.e., not just table of contents or index
    good_chunks = []
    for i, doc in enumerate(docs):
        if len(doc) > 150 and any(w in doc.lower() for w in ["treatment", "diagnosis", "symptoms", "dose", "disease", "infection"]):
            good_chunks.append((metas[i].get('source_file'), doc))
            
    random.seed(42)
    selected = random.sample(good_chunks, min(5, len(good_chunks)))
    
    for i, (source, text) in enumerate(selected):
        print(f"\n[{i+1}] SOURCE: {source}")
        print(f"TEXT: {text}")
except Exception as e:
    print(f"Error: {e}")
