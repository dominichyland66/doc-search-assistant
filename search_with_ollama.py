import requests
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    vec = model.encode(text)
    return np.array(vec, dtype='float32')

def query_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get('response', '').strip()

def main():
    index = faiss.read_index('chunk_index.faiss')
    with open('chunk_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"Loaded index with {len(metadata)} chunks.")

    while True:
        user_question = input("\nEnter your question (or type 'exit' to quit): ").strip()
        if user_question.lower() in ['exit', 'quit']:
            print("Exiting. Goodbye!")
            break

        query_vec = embed_text(user_question)
        D, I = index.search(np.expand_dims(query_vec, axis=0), k=5)

        snippets = []
        for idx in I[0]:
            if idx != -1 and idx < len(metadata):
                file_name, chunk_idx, chunk_text = metadata[idx]
                snippets.append(f"From {file_name}, chunk {chunk_idx}:\n{chunk_text.strip()}")

        if not snippets:
            print("No relevant chunks found.")
            continue

        context = "\n\n".join(snippets)
        prompt = f"""
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{user_question}
"""

        print("\nSending prompt to Ollama...\n")
        answer = query_ollama(prompt)
        print(f"\n🦙 Llama3 Answer:\n{answer}")

if __name__ == "__main__":
    main()
