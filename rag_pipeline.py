"""
rag_pipeline.py  —  FINAL VERSION
====================================
Fixes:
  - "Flasher fold pattern" now correctly returns Flasher document
  - "Star fold" returns Star fold document
  - "Yoshimura" returns Yoshimura document
  - Short queries like "what is satellite" return overview docs
  - No hallucination — Ollama only gets relevant context

Method: BM25 + exact topic name matching boost
"""

import json, math, re, os

KNOWLEDGE_PATH = "knowledge_base.json"
OLLAMA_MODEL   = "llama3"
TOP_K          = 4
K1             = 1.2
B              = 0.5

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","has","have","had",
    "that","this","it","its","as","can","will","also","used","use","uses",
    "using","which","they","their","than","more","most","other","these",
    "those","such","when","where","how","what","who","each","both","all",
    "any","into","not","no","so","if","up","do","did","been","being","just",
    "very","some","about","there","here","then","only","even","still","tell",
    "give","show","explain","describe","define"
}

# Query expansion for short/vague queries
QUERY_EXPAND = {
    "satellite":     "satellite spacecraft orbit types overview",
    "satellites":    "satellite spacecraft orbit types overview",
    "origami":       "origami fold pattern crease deployment",
    "fold":          "origami fold pattern crease mountain valley",
    "solar":         "solar panel array deployment satellite power",
    "antenna":       "antenna deployable reflector satellite communication",
    "cubesat":       "cubesat small satellite deployable nanosatellite",
    "gps":           "navigation satellite GPS GNSS orbit positioning",
    "weather":       "weather satellite meteorological atmospheric monitoring",
    "communication": "communication satellite relay geostationary antenna",
    "telescope":     "space telescope deployable mirror aperture optics",
    "materials":     "materials kapton carbon fiber shape memory origami",
    "deployment":    "deployment mechanism fold unfold satellite structure",
    "miura":         "miura-ori fold pattern satellite solar panel deployment",
    "cvae":          "CVAE ETH Zurich model origami crease pattern generation",
    "eth":           "ETH Zurich CVAE dataset origami satellite",
    "kresling":      "kresling pattern cylindrical deployable bistable",
    "truss":         "truss structure deployable satellite boom",
    "reflector":     "reflector dish antenna satellite deployable",
    "flasher":       "flasher fold pattern rotational circular solar array",
    "yoshimura":     "yoshimura buckling pattern cylindrical deployable boom",
    "waterbomb":     "waterbomb base bistable fold satellite",
    "ikaros":        "IKAROS solar sail JAXA deployment flasher fold",
    "jwst":          "James Webb Space Telescope sunshield origami deployment",
    "iss":           "space station ISS solar array deployable accordion",
    "kapton":        "kapton polyimide film material satellite origami",
    "sma":           "shape memory alloy hinge deployment satellite",
    "gnn":           "graph neural network crease pattern classification",
    "rag":           "retrieval augmented generation knowledge base satellite",
}


# ══════════════════════════════════════════════════════════════════
# TOKENISER
# ══════════════════════════════════════════════════════════════════

def tokenize(text):
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def expand_query(query):
    tokens   = re.findall(r'\b[a-z]+\b', query.lower())
    expanded = query
    for token in tokens:
        if token in QUERY_EXPAND:
            expanded = expanded + " " + QUERY_EXPAND[token]
    return expanded

def topic_similarity(query_tokens, topic):
    """
    How closely does the query match the document topic name?
    Returns a boost score — higher = better topic match.
    """
    topic_tokens = set(tokenize(topic))
    query_set    = set(query_tokens)

    if not topic_tokens:
        return 0.0

    # How many query words appear in topic
    overlap = len(topic_tokens & query_set)

    # Extra boost if ALL topic words are in query
    full_match = 1.0 if topic_tokens.issubset(query_set) else 0.0

    # Extra boost if query is mostly about this topic
    query_coverage = overlap / max(len(query_set), 1)

    return overlap * 3.0 + full_match * 10.0 + query_coverage * 5.0


# ══════════════════════════════════════════════════════════════════
# BM25 INDEX
# ══════════════════════════════════════════════════════════════════

class BM25Index:
    def __init__(self, documents):
        self.docs   = documents
        self.topics = [d.get("topic", "") for d in documents]
        self.N      = len(documents)
        self._build()

    def _build(self):
        self.tokenized = []
        for doc in self.docs:
            # Topic repeated 3x — strong topic signal
            topic_tokens = tokenize(doc.get("topic", "")) * 3
            text_tokens  = tokenize(doc["text"])
            self.tokenized.append(topic_tokens + text_tokens)

        self.avgdl = sum(len(t) for t in self.tokenized) / max(self.N, 1)

        self.df = {}
        for tokens in self.tokenized:
            for word in set(tokens):
                self.df[word] = self.df.get(word, 0) + 1

        self.vectors = []
        for tokens in self.tokenized:
            tf  = {}
            for w in tokens:
                tf[w] = tf.get(w, 0) + 1
            dl  = len(tokens)
            vec = {}
            for word, cnt in tf.items():
                tf_bm25 = (cnt * (K1 + 1)) / (
                    cnt + K1 * (1 - B + B * dl / self.avgdl)
                )
                idf = math.log(
                    (self.N - self.df[word] + 0.5) /
                    (self.df[word] + 0.5) + 1
                )
                vec[word] = tf_bm25 * idf
            self.vectors.append(vec)

    def search(self, query, top_k=TOP_K):
        # Step 1 — expand short queries
        expanded    = expand_query(query)
        qtoks       = tokenize(expanded)
        orig_qtoks  = tokenize(query)   # original query for topic matching

        if not qtoks:
            return []

        scores = []
        for i, vec in enumerate(self.vectors):
            # BM25 text score
            bm25_score = sum(vec.get(w, 0) for w in qtoks)

            # Topic exact match boost — this fixes "Flasher fold pattern"
            topic_boost = topic_similarity(orig_qtoks, self.topics[i])

            total = bm25_score + topic_boost
            scores.append((total, i))

        scores.sort(reverse=True)

        results = []
        for score, i in scores[:top_k]:
            if score < 0.01:
                break
            results.append({
                "text":  self.docs[i]["text"],
                "topic": self.docs[i].get("topic", ""),
                "id":    self.docs[i].get("id", ""),
                "score": round(score, 2)
            })
        return results


# ══════════════════════════════════════════════════════════════════
# OLLAMA
# ══════════════════════════════════════════════════════════════════

SYSTEM = (
    "You are an expert in origami engineering for satellite structures. "
    "Answer using ONLY the context provided below. "
    "Do not use any outside knowledge. "
    "If the context does not fully answer the question, say so clearly. "
    "Be precise, technical, and concise."
)

def ask_ollama(question, chunks):
    context = "\n\n".join(
        f"[Source {i+1} — {c['topic']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    prompt = (
        f"{SYSTEM}\n\n"
        f"### CONTEXT\n{context}\n\n"
        f"### QUESTION\n{question}\n\n"
        f"### ANSWER"
    )
    try:
        import requests
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════
# RAG PIPELINE  — used by step9_backend.py
# ══════════════════════════════════════════════════════════════════

class RAGPipeline:

    def __init__(self, knowledge_path=KNOWLEDGE_PATH):
        if not os.path.exists(knowledge_path):
            raise FileNotFoundError(
                f"knowledge_base.json not found.\n"
                f"Run from the satellite_final folder."
            )
        with open(knowledge_path, encoding="utf-8") as f:
            self.docs = json.load(f)

        self.index = BM25Index(self.docs)
        print(f"[RAG] Ready — {len(self.docs)} documents indexed")

    def retrieve(self, query, top_k=TOP_K):
        return self.index.search(query, top_k=top_k)

    def answer(self, question):
        """Returns (answer_text, sources_list)"""
        chunks = self.retrieve(question)

        if not chunks:
            return "I don't have information about that in my knowledge base.", []

        # Try Ollama
        ollama_answer = ask_ollama(question, chunks)
        if ollama_answer:
            return ollama_answer, chunks

        # Fallback — return best chunk directly
        best = chunks[0]
        return f"[{best['topic']}]\n\n{best['text']}", chunks


# ══════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════

def run_test(rag):
    print()
    print("=" * 65)
    print("  RETRIEVAL TEST")
    print("=" * 65)

    tests = [
        # Exact topic name queries — must return that exact document
        ("Flasher fold pattern",                      "Flasher fold"),
        ("Yoshimura pattern",                         "Yoshimura"),
        ("Waterbomb base",                            "Waterbomb"),
        ("Kresling pattern",                          "Kresling"),
        ("Miura-ori fold",                            "Miura-ori"),
        ("Star fold pattern",                         "Star fold"),
        # Short vague queries — must return relevant overview
        ("what is satellite",                         "satellite"),
        ("what is origami",                           "origami"),
        # Specific domain queries
        ("ETH Zurich CVAE model",                     "CVAE"),
        ("JWST sunshield origami",                    "Webb"),
        ("IKAROS solar sail",                         "IKAROS"),
        ("shape memory alloy hinge",                  "memory alloy"),
        ("CubeSat deployable solar panel",            "CubeSat"),
        ("communication satellite antenna",           "communication"),
    ]

    passed = 0
    for question, must_contain in tests:
        results = rag.retrieve(question, top_k=1)
        if results:
            topic = results[0]["topic"]
            ok    = must_contain.lower() in topic.lower()
            if ok: passed += 1
            status = "PASS ✓" if ok else "FAIL ✗"
            print(f"  {status}  [{results[0]['score']:5.2f}]  {topic}")
            print(f"           Q: {question}")
        else:
            print(f"  FAIL ✗  No result  Q: {question}")
        print()

    print(f"  Result: {passed}/{len(tests)} correct")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 65)
    print("  ORIGAMI SATELLITE RAG  —  Final Version")
    print("  Retrieval : BM25 + Topic Exact Match Boost")
    print("  Generator : Ollama llama3 (if running)")
    print("=" * 65)

    rag = RAGPipeline()
    run_test(rag)

    print("Type your question. Commands: 'sources'  'exit'\n")
    last_sources = []

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "sources":
            if last_sources:
                print("\nSources:")
                for s in last_sources:
                    print(f"  [{s['score']}]  {s['topic']}")
            continue

        answer, last_sources = rag.answer(q)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()