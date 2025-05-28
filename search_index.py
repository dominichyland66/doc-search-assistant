# search_index.py
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
        return None
    start = max(idx - window, 0)
    end = min(idx + len(query) + window, len(text))
    snippet = text[start:end].replace('\n', ' ').strip()
    return f"...{snippet}..."

def main():
    index = faiss.read_index('pdf_index.faiss')
    with open('file_list.pkl', 'rb') as f:
        file_list = pickle.load(f)

    print(f"Loaded index with {len(file_list)} documents.")

    while True:
        query = input("\nEnter a search word or phrase (or type 'exit'): ")
        if query.lower() == 'exit':
            break

        query_vec = embed_text(query)
        D, I = index.search(np.expand_dims(query_vec, axis=0), k=5)

        found = False
        for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
            if idx != -1 and idx < len(file_list):
                matched_file = file_list[idx]
                doc_text = extract_text_from_pdf(matched_file)
                contains_exact = query.lower() in doc_text.lower()

                if contains_exact:
                    snippet = get_snippet(doc_text, query)
                    match_note = f"✅ contains exact match\n   Snippet: {snippet}"
                else:
                    match_note = "⚠️ no exact match"

                print(f"{rank + 1}. {matched_file} (distance {dist:.2f}) — {match_note}")
                found = True

        if not found:
            print("No match found.")

if __name__ == "__main__":
    main()
