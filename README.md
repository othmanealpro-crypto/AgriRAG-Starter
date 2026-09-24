# AgruRAG-Starter

AgriGPT est un assistant agricole intelligent qui permet d'interroger des documents PDF en langage naturel. L'application combine une API FastAPI, une interface Streamlit et une architecture RAG pour retrouver les passages pertinents avant de generer une reponse avec un modele de langage local.

## Fonctionnalites

- Import et analyse de documents PDF
- Extraction et decoupage du contenu en passages
- Recherche semantique avec embeddings et FAISS
- Questions-reponses contextuelles avec un LLM
- Interface web Streamlit
- API backend FastAPI
- Execution locale avec Ollama

## Architecture

```text
Documents PDF
     |
     v
Extraction et decoupage du texte
     |
     v
Embeddings + index FAISS
     |
     v
Recherche des passages pertinents
     |
     v
LLM local via Ollama
     |
     v
Reponse dans Streamlit ou via FastAPI
```

## Technologies

- Python
- FastAPI et Uvicorn
- Streamlit
- FAISS
- Ollama
- LLM et RAG
- Traitement de documents PDF

## Installation

```bash
git clone https://github.com/othmanealpro-crypto/AgruRAG-Starter.git
cd AgruRAG-Starter
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Installer et demarrer Ollama séparément, puis rendre disponible le modele utilise par l'application.

## Lancement

Interface Streamlit :

```bash
streamlit run frontend/app.py
```

API FastAPI :

```bash
uvicorn backend.main:app --reload
```

L'API est ensuite disponible sur `http://127.0.0.1:8000` et sa documentation sur `http://127.0.0.1:8000/docs`.

## Organisation du projet

```text
AgruRAG-Starter/
├── backend/
│   ├── main.py
│   ├── models.py
│   └── rag.py
├── frontend/
│   └── app.py
├── requirements.txt
└── test_*.py
```

## Auteur

Othmane Al Amrani - [GitHub](https://github.com/othmanealpro-crypto)
