from backend.rag import read_pdf_text, split_text_into_chunks

pdf_path = "data/pdfs/FAO_bonnes_pratiques_agricoles.pdf"

text = read_pdf_text(pdf_path)
chunks = split_text_into_chunks(text, chunk_size=500)

print(f"Nombre de chunks : {len(chunks)}\n")
print("Exemple de chunk :\n")
print(chunks[0])
