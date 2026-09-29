"""Simple RAG: chunk -> TF-IDF vectors -> cosine similarity -> cited chunks.
(TF-IDF keeps it dependency-light. Swap in embeddings + Chroma/Qdrant later.)"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_text(text, size=600, overlap=100):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks


def extract_text(uploaded) -> str:
    if uploaded.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(uploaded).pages)
    return uploaded.read().decode("utf-8", errors="ignore")


class DocIndex:
    def __init__(self):
        self.chunks, self.sources, self.vec, self.mat = [], [], None, None

    def add(self, name, text):
        for i, c in enumerate(chunk_text(text)):
            if c.strip():
                self.chunks.append(c)
                self.sources.append(f"{name} (chunk {i + 1})")
        self.vec = TfidfVectorizer(stop_words="english")
        self.mat = self.vec.fit_transform(self.chunks)

    def search(self, query, k=3, min_score=0.08):
        if not self.chunks:
            return []
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.sources[i], "text": self.chunks[i], "score": round(float(sims[i]), 3)}
                for i in top if sims[i] >= min_score]  # empty list => agent must abstain
