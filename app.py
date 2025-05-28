import os
import subprocess
import pickle

import faiss
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify, Blueprint
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Configure Flask:
# - serve CSS/js/etc from 'static/' at '/static'
# - serve templates from 'Templates/'
# - serve PDFs from 'Source_Documents/' at '/Source_Documents'
# ----------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder='Templates',
    static_folder='static',
    static_url_path='/static'
)

# Blueprint to serve PDF files as a second "static" path
pdf_bp = Blueprint(
    'pdf',
    __name__,
    static_folder='Source_Documents',
    static_url_path='/Source_Documents'
)
app.register_blueprint(pdf_bp)

# ----------------------------------------------------------------------------
# Constants & Model Initialization
# ----------------------------------------------------------------------------
OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"
model = SentenceTransformer('all-MiniLM-L6-v2')

# Index & Metadata placeholders
index = None
metadata = None

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
def load_index():
    global index, metadata
    # Load FAISS index
    index = faiss.read_index('chunk_index.faiss')
    # Load metadata mapping each vector to (filename, chunk_idx, text)
    with open('chunk_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)


def embed_text(text: str) -> np.ndarray:
    """Embed a text string into a float32 vector."""
    vec = model.encode(text)
    return np.array(vec, dtype='float32')


def query_ollama(prompt: str) -> str:
    """Send a prompt to the local Ollama API and return the text response."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    return resp.json().get('response', '').strip()

# ----------------------------------------------------------------------------
# Flask Routes
# ----------------------------------------------------------------------------
@app.route('/')
def index_page():
    """Render the main search UI."""
    return render_template('index.html')


@app.route('/list_files', methods=['GET'])
def list_files():
    """Return a JSON list of all PDFs in Source_Documents/"""
    source_folder = 'Source_Documents'
    try:
        files = [f for f in os.listdir(source_folder) if f.lower().endswith('.pdf')]
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/reindex', methods=['POST'])
def reindex():
    """Rebuild the FAISS index by invoking build_index.py."""
    subprocess.run(['python', 'build_index.py'], check=True)
    load_index()
    return jsonify({"message": "Index rebuilt successfully."})


@app.route('/search', methods=['POST'])
def search():
    """Handle a user query: retrieve top chunks, send context to Ollama, return answer."""
    user_question = request.form.get('query', '')
    if not user_question:
        return jsonify({"answer": "Please provide a query."}), 400

    # Embed and search
    q_vec = embed_text(user_question)
    distances, indices = index.search(np.expand_dims(q_vec, axis=0), k=5)

    snippets = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            fname, chunk_id, text = metadata[idx]
            snippets.append(f"From {fname}, chunk {chunk_id}:\n{text.strip()}")
    # something
    if not snippets:
        return jsonify({"answer": "No relevant chunks found."})

    context = "\n\n".join(snippets)
    prompt = f"""
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{user_question}
"""
    answer = query_ollama(prompt)
    return jsonify({"answer": answer})

# ----------------------------------------------------------------------------
# Application Entry Point
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    load_index()
    app.run(host='0.0.0.0', port=5000)
