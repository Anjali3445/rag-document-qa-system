"""Load text from .txt, .md, .pdf and .docx files."""
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def read_file(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if ext == ".docx":
        import docx
        return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    raise ValueError(f"Unsupported file type: {ext}")


def load_documents(paths):
    """paths: files or folders. Returns list of {'source', 'text'}."""
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.suffix.lower() in SUPPORTED]
        elif p.suffix.lower() in SUPPORTED:
            files.append(p)
    docs = []
    for f in files:
        text = read_file(f).strip()
        if text:
            docs.append({"source": f.name, "text": text})
    return docs
