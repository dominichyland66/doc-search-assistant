# step1_search_snippets.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import fitz  # PyMuPDF

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    vec = model.encode(text)
    return np.array(vec, dtype='float32')

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def get_snippet(text, query, window=50):
    text_lower = text.lower()
    query_lower = query.lower()
    idx = text_lower.find(query_lower)
    if idx == -1:
        # fallback: return start of document
        snippet = text[:window * 2].replace('\n', ' ').strip()
    else:
        start = max(idx - window, 0)
        end = min(idx + len(query) + window, len(text))
        snippet = text[start:end].replace('\n', ' ').strip()
    return f"...{snippet}..."

def main():
    index = faiss.read_index('pdf_index.faiss')
    with open('file_list.pkl', 'rb') as f:
        file_list = pickle.load(f)

    print(f"Loaded index with {len(file_list)} documents.")

    query = input("\nEnter a search question: ")
    query_vec = embed_text(query)

    D, I = index.search(np.expand_dims(query_vec, axis=0), k=3)

    print("\nTop 3 matching documents:\n")
    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        if idx != -1 and idx < len(file_list):
            matched_file = file_list[idx]
            doc_text = extract_text_from_pdf(matched_file)
            snippet = get_snippet(doc_text, query)
            print(f"{rank + 1}. {matched_file} (distance {dist:.2f})")
            print(f"   Snippet: {snippet}\n")

if __name__ == "__main__":
    main()
