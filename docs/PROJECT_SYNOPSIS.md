# Project Synopsis: RAG-Based Document QA System

**1. Problem statement** – Employees waste time searching long documents for answers. LLMs alone
can hallucinate and do not know private documents.

**2. Objective** – Build a system that answers questions strictly from user-supplied documents and cites sources.

**3. Approach (RAG)** – Load documents -> split into overlapping chunks -> index with TF-IDF or sentence
embeddings -> retrieve top-k chunks for a question -> generate an answer with an LLM (or extractive fallback).

**4. Dataset** – Two fictional policy documents plus a 10-question evaluation set with ground truth
(see `data/DATA_DICTIONARY.md`). Generated using `docs/DATASET_GENERATION_PROMPT.md`.

**5. Tools** – Python, scikit-learn, Streamlit, pandas, optional sentence-transformers / Claude / OpenAI.

**6. Evaluation** – Hit@k (is a correct chunk retrieved?) and MRR (how high is it ranked?).
Result on sample data: Hit@3 = 100%, MRR = 1.0.

**7. Deliverables** – Source code, web app, CLI, dataset, data dictionary, evaluation results, README.

**8. Limitations and future work** – TF-IDF is keyword-based (use embeddings for paraphrases);
no OCR for scanned PDFs; add re-ranking, conversation memory and a vector database (FAISS/Chroma).
