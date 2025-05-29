#!/usr/bin/env python3
# build_graph_with_lda.py

import os
import json
import pickle
import argparse

import numpy as np
import faiss

import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

def main():
    p = argparse.ArgumentParser(
        description="Build a D3 graph.json from PDF chunks via LDA topics"
    )
    p.add_argument(
        '--num-topics', '-t', type=int, default=20,
        help="Number of LDA topics (fallback if no topics-json provided)"
    )
    p.add_argument(
        '--topics-json', '-j', default=None,
        help="Path to a JSON list of seed topics; its length will override --num-topics"
    )
    p.add_argument(
        '--top-n-keywords', '-k', type=int, default=5,
        help="Number of top keywords to extract per topic"
    )
    p.add_argument(
        '--k-neighbors', '-n', type=int, default=5,
        help="k for k-NN when building links"
    )
    p.add_argument(
        '--data-dir', '-d', default='.',
        help="Where chunk_index.faiss & chunk_metadata.pkl live"
    )
    p.add_argument(
        '--static-dir', '-s', default='static',
        help="Where to write graph.json & lda_topics.json"
    )
    args = p.parse_args()

    # --- 1) Load seed topics (if any) to pull descriptions ---
    seed_desc = {}
    if args.topics_json and os.path.exists(args.topics_json):
        with open(args.topics_json, 'r') as f:
            seed = json.load(f)
        # expect entries like {id:0, keywords:[...], description:"..."}
        for entry in seed:
            tid = entry.get("id")
            seed_desc[tid] = entry.get("description", "")
        NUM_TOPICS = len(seed)
        print(f"[init] Loaded {NUM_TOPICS} seed topics (with descriptions) from {args.topics_json}")
    else:
        NUM_TOPICS = args.num_topics
        if args.topics_json:
            print(f"[warning] topics-json not found at {args.topics_json}, using --num-topics")
        print(f"[init] Using --num-topics = {NUM_TOPICS}")

    TOP_N_KEYWORDS = args.top_n_keywords
    K_NEIGHBORS    = args.k_neighbors
    DATA_DIR       = args.data_dir
    STATIC_DIR     = args.static_dir

    # --- 2) Load FAISS index & metadata ---
    idx = faiss.read_index(os.path.join(DATA_DIR, 'chunk_index.faiss'))
    with open(os.path.join(DATA_DIR, 'chunk_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    n_chunks = len(metadata)
    print(f"[1] Loaded {n_chunks} chunks")

    # --- 3) Preprocess texts for LDA ---
    texts = []
    for _, _, txt in metadata:
        tokens = [
            token for token in simple_preprocess(txt, deacc=True)
            if token not in STOPWORDS and len(token) > 3
        ]
        texts.append(tokens)

    # --- 4) Build dictionary & corpus ---
    dictionary = corpora.Dictionary(texts)
    corpus     = [dictionary.doc2bow(t) for t in texts]

    # --- 5) Train LDA model ---
    print(f"[2] Training LDA with {NUM_TOPICS} topics…")
    lda = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=10,
        random_state=42
    )

    # --- 6) Extract topic → keywords & assemble descriptions ---
    lda_topics = {}
    for tid in range(NUM_TOPICS):
        kws = [word for word, _ in lda.show_topic(tid, TOP_N_KEYWORDS)]
        lda_topics[tid] = kws

    # --- 7) Write out lda_topics.json with descriptions ---
    os.makedirs(STATIC_DIR, exist_ok=True)
    topics_out = []
    for tid, kws in lda_topics.items():
        topics_out.append({
            "id":          tid,
            "keywords":    kws,
            "description": seed_desc.get(tid, "")
        })
    with open(os.path.join(STATIC_DIR, 'lda_topics.json'), 'w') as f:
        json.dump(topics_out, f, indent=2)
    print(f"[3] Wrote {len(topics_out)} topics (+descriptions) → {STATIC_DIR}/lda_topics.json")

    # --- 8) Assign each chunk its dominant topic ---
    chunk_topics = []
    for bow in corpus:
        dist = lda.get_document_topics(bow)
        top_topic = max(dist, key=lambda x: x[1])[0]
        chunk_topics.append(top_topic)

    # --- 9) Build k-NN links ---
    all_vecs = idx.reconstruct_n(0, n_chunks)
    D, I     = idx.search(all_vecs, K_NEIGHBORS + 1)
    links = []
    for src, (nbrs, dists) in enumerate(zip(I, D)):
        for dst, dist in zip(nbrs[1:], dists[1:]):
            links.append({
                "source": src,
                "target": int(dst),
                "value": 1.0 / (1.0 + dist)
            })

    # --- 10) Build graph nodes & dump graph.json ---
    nodes = []
    for i, (fname, chunk_id, _) in enumerate(metadata):
        tid = chunk_topics[i]
        nodes.append({
            "id":       i,
            "label":    f"{fname}-{chunk_id}",
            "topic":    tid,
            "keywords": lda_topics[tid],
            # you could also include description here if desired:
            # "description": seed_desc.get(tid, "")
        })

    graph_out = {"nodes": nodes, "links": links}
    with open(os.path.join(STATIC_DIR, 'graph.json'), 'w') as f:
        json.dump(graph_out, f, indent=2)
    print(f"[4] Exported graph with {len(nodes)} nodes & {len(links)} links → {STATIC_DIR}/graph.json")


if __name__ == "__main__":
    main()
