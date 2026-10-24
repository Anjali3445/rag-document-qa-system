"""Answer generation. Uses an LLM if an API key exists, else an extractive fallback."""
import os
import re
from .textutil import tokenize

PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find that in the documents."
Keep the answer concise and mention the source file names you used.

Context:
{context}

Question: {question}
Answer:"""


def _tokens(s):
    return set(tokenize(s))


def extractive_answer(question, chunks, n_sentences=2):
    """No-LLM fallback: return the sentences most similar to the question."""
    q = _tokens(question)
    cands = []
    for rank, ch in enumerate(chunks):
        for s in re.split(r"(?<=[.!?])\s+", ch["text"]):
            overlap = len(q & _tokens(s))
            if overlap:
                cands.append((overlap - 0.01 * rank, s.strip()))
    if not cands:
        return "I could not find that in the documents."
    best = sorted(cands, key=lambda x: -x[0])[:n_sentences]
    return " ".join(s for _, s in best)


def llm_answer(question, chunks):
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    prompt = PROMPT.format(context=context, question=question)
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        r = client.messages.create(
            model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content
    return None


def generate(question, chunks, use_llm=True):
    if not chunks:
        return "I could not find that in the documents.", "none"
    if use_llm:
        try:
            ans = llm_answer(question, chunks)
            if ans:
                return ans, "llm"
        except Exception as e:  # network/key problems -> fall back
            print(f"[warn] LLM failed ({e}); using extractive answer")
    return extractive_answer(question, chunks), "extractive"
