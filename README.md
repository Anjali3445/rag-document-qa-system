# RAG-Based Document QA System

Ask questions about your own documents (TXT, MD, PDF, DOCX). The system retrieves the most
relevant passages and answers **only from them**, showing the sources.

## Run in 3 steps
```bash
pip install -r requirements.txt          # 1. install
streamlit run app.py                     # 2a. web app  (opens in browser)
python cli.py                            # 2b. or terminal chat
python evaluate.py                       # 3. measure retrieval quality (Hit@k, MRR)
```
Works out of the box: **no API key, no model download** (TF-IDF retrieval + extractive answers).

## Optional upgrades
| Want | Do |
|---|---|
| Better semantic retrieval | `pip install sentence-transformers`, choose "embeddings" in the app (or `--backend embeddings`) |
| Fluent LLM answers | `pip install anthropic` and set `ANTHROPIC_API_KEY` (or `pip install openai` + `OPENAI_API_KEY`). Change model with `LLM_MODEL` |

## Architecture
```
Documents -> Loader -> Chunker (120 words, 30 overlap) -> Retriever index (TF-IDF / embeddings)
Question  -> Retriever (top-k chunks) -> Generator (LLM or extractive) -> Answer + Sources
```

## Folder structure
```
app.py                Streamlit UI (upload, settings, chat, sources)
cli.py                Terminal interface
evaluate.py           Hit@k and MRR on data/qa_dataset.csv -> outputs/evaluation_results.csv
rag/                  loader.py, chunker.py, retriever.py, generator.py, pipeline.py
data/documents/       Sample corpus (2 policy documents)
data/qa_dataset.csv   10 evaluation questions with ground truth
data/DATA_DICTIONARY.md
docs/                 PROJECT_SYNOPSIS.md, DATASET_GENERATION_PROMPT.md
```

## Results (sample dataset, top-3, TF-IDF)
Hit@3 = 100%, MRR = 1.0 (run `python evaluate.py` to reproduce).

## Tech stack
Python, scikit-learn, pandas, Streamlit, pypdf, python-docx; optional sentence-transformers, Anthropic/OpenAI APIs.
