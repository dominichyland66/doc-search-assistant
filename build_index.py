import os
import pickle
import numpy as np
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_chunks_from_pdf(pdf_path, chunk_size=500):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def embed_text(text):
    vec = model.encode(text)
    return np.array(vec, dtype='float32')

def main():
    embedding_dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(embedding_dim)
    metadata = []  # store (filename, chunk index, chunk text)

    source_folder = "Source_Documents"

    for file in os.listdir(source_folder):
        if file.endswith('.pdf'):
            file_path = os.path.join(source_folder, file)
            print(f"Processing {file_path}...")
            chunks = extract_chunks_from_pdf(file_path)
            for i, chunk in enumerate(chunks):
                vec = embed_text(chunk)
                index.add(np.expand_dims(vec, axis=0))
                metadata.append((file, i, chunk))

    faiss.write_index(index, 'chunk_index.faiss')
    with open('chunk_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Saved index with {len(metadata)} chunks.")

if __name__ == "__main__":
    main()
