import requests
from backend.rag import read_pdf_text, split_text_into_chunks

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "mxbai-embed-large"

pdf_path = "data/pdfs/FAO_bonnes_pratiques_agricoles.pdf"

# 1. Lire le PDF
text = read_pdf_text(pdf_path)

# 2. Découper en chunks
chunks = split_text_into_chunks(text, chunk_size=500)

print(f"Nombre de chunks : {len(chunks)}")

# 3. Générer les embeddings
embeddings = []

for i, chunk in enumerate(chunks):
    payload = {
        "model": MODEL_NAME,
        "prompt": chunk
    }
    response = requests.post(OLLAMA_URL, json=payload)
    embedding = response.json()["embedding"]
    embeddings.append(embedding)

    print(f"Chunk {i+1}/{len(chunks)} → embedding taille {len(embedding)}")

print("\n✅ Tous les embeddings ont été générés.")
