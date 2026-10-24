"""Split documents into overlapping word-based chunks (paragraph aware)."""
import re


def chunk_text(text, chunk_size=120, overlap=30):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    words = []
    for p in paragraphs:
        words.extend(p.split())
    chunks, start = [], 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        piece = words[start:start + chunk_size]
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def chunk_documents(docs, chunk_size=120, overlap=30):
    out = []
    for d in docs:
        for i, c in enumerate(chunk_text(d["text"], chunk_size, overlap)):
            out.append({"source": d["source"], "chunk_id": i, "text": c})
    return out
