"""Evaluate retrieval on data/qa_dataset.csv -> Hit@k and MRR, saved to outputs/."""
import os
import pandas as pd
from rag import RAGPipeline

K = 3
rag = RAGPipeline(k=K, use_llm=False)
rag.ingest(["data/documents"])
df = pd.read_csv("data/qa_dataset.csv")

rows, hits, rr = [], 0, 0.0
for _, r in df.iterrows():
    res = rag.retrieve(r["question"])
    rank = next((i + 1 for i, c in enumerate(res)
                 if r["source_file"] == c["source"] and r["answer_keyword"].lower() in c["text"].lower()), None)
    hits += rank is not None
    rr += 1 / rank if rank else 0
    rows.append({"id": r["id"], "question": r["question"], "hit": rank is not None, "rank": rank})

n = len(df)
print(f"Questions: {n} | Hit@{K}: {hits / n:.2%} | MRR: {rr / n:.3f}")
os.makedirs("outputs", exist_ok=True)
pd.DataFrame(rows).to_csv("outputs/evaluation_results.csv", index=False)
print("Saved outputs/evaluation_results.csv")
