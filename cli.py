"""Command-line chat:  python cli.py --docs data/documents"""
import argparse
from rag import RAGPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--docs", nargs="+", default=["data/documents"])
ap.add_argument("--backend", choices=["tfidf", "embeddings"], default="tfidf")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--no-llm", action="store_true")
ap.add_argument("-q", "--question", help="Ask a single question and exit")
a = ap.parse_args()

rag = RAGPipeline(backend=a.backend, k=a.k, use_llm=not a.no_llm)
print("Indexed:", rag.ingest(a.docs))


def show(q):
    r = rag.ask(q)
    print(f"\nAnswer ({r['mode']}): {r['answer']}\n")
    for s in r["sources"]:
        print(f"  - {s['source']} (chunk {s['chunk_id']}, score {s['score']})")


if a.question:
    show(a.question)
else:
    print("Type a question (or 'exit').")
    while (q := input("\n> ").strip()).lower() not in {"exit", "quit"}:
        if q:
            show(q)
