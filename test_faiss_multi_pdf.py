import requests
import numpy as np
import faiss

from backend.rag import load_pdfs_and_create_chunks

# --- Config ---
PDF_DIR = "data/pdfs"
EMBED_MODEL = "mxbai-embed-large"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

# --- Embedding helper ---
def embed_text(text: str):
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    res = requests.post(OLLAMA_EMBED_URL, json=payload)
    return res.json()["embedding"]

# --- 1. Charger chunks multi-PDF ---
chunks = load_pdfs_and_create_chunks(PDF_DIR)
print(f"Nombre total de chunks : {len(chunks)}")

# --- 2. Générer embeddings ---
embeddings = []
for chunk in chunks:
    embeddings.append(embed_text(chunk["text"]))

embeddings = np.array(embeddings).astype("float32")
dim = embeddings.shape[1]

# --- 3. Index FAISS ---
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print("Index FAISS prêt. Vecteurs stockés :", index.ntotal)

# --- 4. Recherche ---
question = "Comment bien gérer l’irrigation du blé ?"
question_embedding = np.array([embed_text(question)]).astype("float32")

k = 5
distances, indices = index.search(question_embedding, k)

print("\nTop chunks pertinents (avec sources) :\n")
for rank, idx in enumerate(indices[0]):
    chunk = chunks[idx]
    print(f"--- Rang {rank+1} | Source : {chunk['source']} | Distance : {distances[0][rank]:.2f}")
    print(chunk["text"][:400])
    print()
