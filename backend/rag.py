import requests
import numpy as np
import faiss
from pathlib import Path
from pypdf import PdfReader

# --- Config ---
PDF_DIR = "data/pdfs"
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2:3b"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

# --- Variables globales (initialisées plus tard) ---
index = None
chunks = []


# ---------- Utils ----------
def read_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def split_text_into_chunks(text: str, chunk_size: int = 500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]


def embed_text(text: str):
    payload = {"model": EMBED_MODEL, "prompt": text}
    res = requests.post(OLLAMA_EMBED_URL, json=payload)
    return res.json()["embedding"]


# ---------- INITIALISATION CONTRÔLÉE ----------
def init_rag():
    global index, chunks

    if index is not None:
        return  # déjà initialisé

    print("🔄 Initialisation du moteur RAG...")

    chunks = []
    for pdf in Path(PDF_DIR).glob("*.pdf"):
        text = read_pdf_text(str(pdf))
        for chunk in split_text_into_chunks(text):
            chunks.append({
                "text": chunk,
                "source": pdf.name
            })

    embeddings = np.array([embed_text(c["text"]) for c in chunks]).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("✅ RAG initialisé avec", len(chunks), "chunks.")


# ---------- RAG ----------
def ask_question(question: str, k: int = 5):
    # 0) Filtre "domaine agricole" (simple et efficace pour un PFA)
    agriculture_keywords = [
        "agriculture", "agricole", "blé", "orge", "maïs", "culture", "cultures",
        "irrigation", "eau", "sol", "fertilisation", "engrais", "semis",
        "récolte", "rendement", "maladie", "ravageur", "pesticide", "herbicide",
        "désherbage", "phytosanitaire", "rotation", "semence", "variété",
        "élevage", "bétail", "fourrage", "serre"
    ]

    q_lower = question.lower()
    if not any(kw in q_lower for kw in agriculture_keywords):
        return (
            "Je ne peux répondre qu’aux questions agricoles basées sur les documents fournis. "
            "Votre question semble hors du domaine agricole.",
            []
        )

    # 1) Init RAG si nécessaire
    if index is None:
        init_rag()

    # 2) Recherche FAISS
    q_emb = np.array([embed_text(question)]).astype("float32")
    distances, indices = index.search(q_emb, k)

    # 3) Seuil de pertinence : si trop loin => "je ne sais pas"
    # (seuil empirique: à ajuster si besoin)
    best_distance = float(distances[0][0])
    DISTANCE_THRESHOLD = 190.0

    if best_distance > DISTANCE_THRESHOLD:
        return (
            "Je ne trouve pas d’information pertinente dans les documents fournis pour répondre à cette question.",
            []
        )

    retrieved_chunks = [chunks[i] for i in indices[0]]

    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = f"""
Tu es un assistant agricole intelligent.
Réponds uniquement à partir des informations ci-dessous.
Si l'information n'est pas présente, dis-le clairement.

Informations :
{context}

Question : {question}
Réponse :
"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    res = requests.post(OLLAMA_GENERATE_URL, json=payload)
    answer = res.json()["response"]

    sources = list({c["source"] for c in retrieved_chunks})
    return answer, sources
