from .loader import load_documents
from .chunker import chunk_documents
from .retriever import make_retriever, top_k
from .generator import generate


class RAGPipeline:
    def __init__(self, backend="tfidf", chunk_size=120, overlap=30, k=3, use_llm=True):
        self.retriever = make_retriever(backend)
        self.chunk_size, self.overlap, self.k, self.use_llm = chunk_size, overlap, k, use_llm
        self.chunks = []

    def ingest(self, paths):
        docs = load_documents(paths)
        if not docs:
            raise ValueError("No readable documents found (supported: txt, md, pdf, docx).")
        self.chunks = chunk_documents(docs, self.chunk_size, self.overlap)
        self.retriever.fit([c["text"] for c in self.chunks])
        return {"documents": len(docs), "chunks": len(self.chunks)}

    def retrieve(self, question):
        hits = top_k(self.retriever.scores(question), self.k)
        return [{**self.chunks[i], "score": round(s, 3)} for i, s in hits]

    def ask(self, question):
        hits = self.retrieve(question)
        answer, mode = generate(question, hits, self.use_llm)
        return {"question": question, "answer": answer, "mode": mode,
                "retriever": self.retriever.name, "sources": hits}
