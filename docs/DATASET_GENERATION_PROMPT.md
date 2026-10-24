# Dataset Generation Prompt (paste into any LLM)

You are creating a dataset for a RAG Document QA system.
Write 3 realistic company policy documents (~400 words each, plain paragraphs
with specific numbers, durations and rules). Then produce a CSV with columns:
id, question, ground_truth_answer, source_file, answer_keyword, category.
Rules: 20 questions total; each answer must be explicitly stated in exactly one
document; answer_keyword must be an exact phrase from that document;
questions must be phrased differently from the document text.
