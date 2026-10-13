from pathlib import Path
import os, uuid
from dotenv import load_dotenv
load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "chroma_db"
KNOW = ROOT / "knowledge"
_collection = None

def chunk(text, size=850, overlap=120):
    text = " ".join(text.split())
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        out.append(text[start:end])
        if end == len(text): break
        start = end - overlap
    return out

def collection():
    global _collection
    if _collection is not None: return _collection
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    client = chromadb.PersistentClient(path=str(DB))
    ef = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _collection = client.get_or_create_collection("estateiq_knowledge", embedding_function=ef, metadata={"hnsw:space":"cosine"})
    if _collection.count() == 0:
        docs, metas, ids = [], [], []
        for p in KNOW.iterdir():
            if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
                for i, c in enumerate(chunk(p.read_text(errors="ignore"))):
                    docs.append(c); metas.append({"source":p.name,"chunk":i+1}); ids.append(f"{p.stem}-{i}-{uuid.uuid4().hex[:6]}")
        if docs: _collection.add(documents=docs, metadatas=metas, ids=ids)
    return _collection

def index_file(filename, raw):
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt",".md",".markdown",".pdf"}: raise ValueError("Use PDF, TXT, or Markdown.")
    if suffix == ".pdf":
        from pypdf import PdfReader
        import io
        text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
    else: text = raw.decode("utf-8", errors="ignore")
    parts = chunk(text)
    if not parts: raise ValueError("No readable text in this file.")
    col = collection()
    col.add(documents=parts, metadatas=[{"source":Path(filename).name,"chunk":i+1} for i in range(len(parts))],
            ids=[f"up-{uuid.uuid4().hex}" for _ in parts])
    return len(parts)

def retrieve(question, k=3):
    col = collection()
    if not col.count(): return []
    r = col.query(query_texts=[question], n_results=min(k,col.count()))
    return [{"text":d,"source":m.get("source","unknown"),"chunk":m.get("chunk","?"),"distance":round(float(r["distances"][0][i]),4)}
            for i,(d,m) in enumerate(zip(r["documents"][0],r["metadatas"][0]))]

def ask(question):
    sources = retrieve(question, 3)
    context = "\n\n".join(f"[{x['source']} chunk {x['chunk']}]\n{x['text']}" for x in sources)
    key = os.getenv("GROQ_API_KEY","").strip()
    if not key:
        return {"answer":"RAG retrieval worked, but the LLM is disabled. Add GROQ_API_KEY to .env to generate an AI answer.",
                "sources":sources,"mode":"retrieval-only"}
    from groq import Groq
    client = Groq(api_key=key)
    prompt = f"""You are EstateIQ, a careful real-estate research assistant. Answer only from the retrieved context. If it does not answer the question, say the evidence is insufficient. Do not invent current prices, listings, legal advice, or guarantees. Mention source filenames where useful.

CONTEXT:
{context or "No relevant context found."}

QUESTION:
{question}"""
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL","llama-3.3-70b-versatile"),
        messages=[{"role":"system","content":"Be accurate, concise, and transparent about uncertainty."},
                  {"role":"user","content":prompt}], temperature=0.2, max_tokens=650)
    return {"answer":response.choices[0].message.content,"sources":sources,"mode":"groq-rag"}
