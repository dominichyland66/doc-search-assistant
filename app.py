import os
import subprocess
import pickle
import json

import faiss
import numpy as np
import requests
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Blueprint,
    send_from_directory,
)
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Configure Flask
# ----------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder='Templates',
    static_folder='static',
    static_url_path='/static'
)

pdf_bp = Blueprint(
    'pdf',
    __name__,
    static_folder='Source_Documents',
    static_url_path='/Source_Documents'
)
app.register_blueprint(pdf_bp)

# ----------------------------------------------------------------------------
# Globals & Model
# ----------------------------------------------------------------------------
OLLAMA_URL   = "http://host.docker.internal:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"
model        = SentenceTransformer('all-MiniLM-L6-v2')

index    = None
metadata = None

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_index():
    global index, metadata
    index = faiss.read_index('chunk_index.faiss')
    with open('chunk_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

def embed_text(text: str) -> np.ndarray:
    return np.array(model.encode(text), dtype='float32')

def query_ollama(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )
    resp.raise_for_status()
    return resp.json().get('response', '').strip()

# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/graph')
def graph_viewer():
    return send_from_directory(app.static_folder, 'graph.html')

@app.route('/graph_data')
def graph_data():
    path = os.path.join(app.static_folder, 'graph.json')
    with open(path, 'r') as f:
        return jsonify(json.load(f))

@app.route('/list_files', methods=['GET'])
def list_files():
    try:
        files = [
            f for f in os.listdir('Source_Documents')
            if f.lower().endswith('.pdf')
        ]
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reindex', methods=['POST'])
def reindex():
    subprocess.run(['python', 'build_index.py'], check=True)
    load_index()
    return jsonify({"message": "Index rebuilt successfully."})

@app.route('/search', methods=['POST'])
def search():
    q = request.form.get('query', '').strip()
    if not q:
        return jsonify({"answer": "Please provide a query."}), 400

    qv = embed_text(q)
    D, I = index.search(np.expand_dims(qv, 0), k=5)

    snippets = []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            fn, cid, txt = metadata[idx]
            snippets.append(f"From {fn} (chunk {cid}):\n{txt.strip()}")

    if not snippets:
        return jsonify({"answer": "No relevant chunks found."})

    context = "\n\n".join(snippets)
    prompt = (
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n" + context + "\n\n"
        "Question:\n" + q
    )

    answer = query_ollama(prompt)
    return jsonify({"answer": answer})

@app.route('/chunk/<int:node_id>')
def get_chunk(node_id):
    """Return the filename, chunk index, and text for a given node ID."""
    if 0 <= node_id < len(metadata):
        filename, chunk_idx, text = metadata[node_id]
        return jsonify({
            "filename": filename,
            "chunk_idx": chunk_idx,
            "text": text
        })
    else:
        return jsonify({"error": "Invalid node id"}), 404

@app.route('/topic_chunks/<int:topic_id>')
def topic_chunks(topic_id):
    """Return all chunks (with excerpts) for a given topic."""
    # Load graph.json to find which nodes have this topic
    graph_path = os.path.join(app.static_folder, 'graph.json')
    with open(graph_path, 'r') as f:
        graph = json.load(f)
    ids = [node['id'] for node in graph['nodes'] if node['topic'] == topic_id]

    result = []
    for idx in ids:
        fn, cid, txt = metadata[idx]
        excerpt = txt.strip().replace('\n', ' ')[:300] + "…"
        result.append({
            "chunk_id":    idx,
            "filename":    fn,
            "chunk_index": cid,
            "excerpt":     excerpt
        })
    return jsonify(result)

# ----------------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    load_index()
    app.run(host='0.0.0.0', port=5000, debug=True)
