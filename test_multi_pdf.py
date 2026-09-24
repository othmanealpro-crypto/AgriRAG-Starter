from backend.rag import load_pdfs_and_create_chunks

chunks = load_pdfs_and_create_chunks("data/pdfs")

print(f"Nombre total de chunks : {len(chunks)}\n")

print("Exemples de chunks avec source :\n")

for i in range(3):
    print(f"--- Chunk {i+1} ---")
    print("Source :", chunks[i]["source"])
    print(chunks[i]["text"][:300])
    print()
