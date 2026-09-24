import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "mxbai-embed-large"

text = "L'irrigation est une pratique essentielle pour la culture du blé."

payload = {
    "model": MODEL_NAME,
    "prompt": text
}

response = requests.post(OLLAMA_URL, json=payload)
data = response.json()

embedding = data["embedding"]

print("Taille de l'embedding :", len(embedding))
print("Extrait de l'embedding :", embedding[:10])
