import requests
import numpy as np
import faiss

from backend.rag import read_pdf_text, split_text_into_chunks

# --- Config ---
PDF_PATH = "data/pdfs/FAO_bonnes_pratiques_agricoles.pdf"
EMBED_MODEL = "mxbai-embed-large"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

# --- Fonctions ---
def embed_text(text: str):
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    res = requests.post(OLLAMA_EMBED_URL, json=payload)
    return res.json()["embedding"]

# --- 1. Lire le PDF ---
text = read_pdf_text(PDF_PATH)

# --- 2. Découper en chunks ---
chunks = split_text_into_chunks(text, chunk_size=500)
print(f"Nombre de chunks : {len(chunks)}")

# --- 3. Générer embeddings ---
embeddings = []
for chunk in chunks:
    embeddings.append(embed_text(chunk))

embeddings = np.array(embeddings).astype("float32")
dim = embeddings.shape[1]
print("Dimension des embeddings :", dim)

# --- 4. Créer l’index FAISS ---
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print("Index FAISS créé. Vecteurs stockés :", index.ntotal)

# --- 5. Tester une recherche ---
question = "Comment bien gérer l’irrigation des cultures ?"
question_embedding = np.array([embed_text(question)]).astype("float32")

k = 3
distances, indices = index.search(question_embedding, k)

print("\nTop chunks pertinents :\n")
for i, idx in enumerate(indices[0]):
    print(f"--- Chunk {i+1} (distance {distances[0][i]:.2f}) ---")
    print(chunks[idx][:400])
    print()
