import json
from pathlib import Path

import chromadb
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from retriever import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def load_pdf(path: Path) -> list[dict]:
    """Return one {"text", "page"} dict per non-empty page."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "page": i})
    return pages


def chunk_text_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size character windows with overlap."""
    step = chunk_size - overlap
    return [text[i : i + chunk_size] for i in range(0, len(text), step) if text[i : i + chunk_size].strip()]


def chunk_text_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split on paragraphs, then lines, sentences and words, only cutting mid-word as a last resort."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Chunk text with the strategy chosen in config.CHUNKING."""
    if config.CHUNKING == "recursive":
        return chunk_text_recursive(text, chunk_size, overlap)
    return chunk_text_fixed(text, chunk_size, overlap)


def load_articles(path: Path) -> list[dict]:
    """Load MultiHop-RAG articles; the url is the document id."""
    docs = []
    for a in json.loads(Path(path).read_text()):
        header = f"{a['title']} | {a['source']} | {a['published_at'][:10]}"
        docs.append(
            {
                "doc_id": a["url"],
                "text": a["body"],
                "header": header if config.ARTICLE_HEADER else "",
                "metadata": {"title": a["title"], "source": a["source"], "published_at": a["published_at"]},
            }
        )
    return docs


def load_documents() -> list[dict]:
    """Return one {"doc_id", "text", "metadata"} dict per document of the configured DATASET."""
    if config.DATASET == "multihop":
        return load_articles(Path(config.MULTIHOP_DIR) / "corpus.json")
    docs = []
    for pdf in sorted(Path(config.DATA_DIR).glob("*.pdf")):
        for page in load_pdf(pdf):
            docs.append(
                {
                    "doc_id": f"RAG-{page['page']:03d}",  # one section per page
                    "text": page["text"],
                    "metadata": {"source": pdf.name, "page": page["page"]},
                }
            )
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Chunk each document, keeping its metadata, doc_id and a stable chunk id."""
    chunks = []
    for doc in docs:
        for j, text in enumerate(chunk_text(doc["text"])):
            header = doc.get("header")
            chunks.append(
                {
                    "id": f"{doc['doc_id']}:c{j}",
                    "text": f"{header}\n{text}" if header else text,
                    "metadata": {**doc["metadata"], "doc_id": doc["doc_id"], "chunk": j},
                }
            )
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with OpenAI, in batches."""
    embeddings = []
    for i in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[i : i + config.EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in resp.data)
    return embeddings


def get_collection(reset: bool = False):
    """Open (or recreate) the persistent Chroma collection."""
    db = chromadb.PersistentClient(path=config.CHROMA_DIR)
    if reset:
        try:
            db.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass
    return db.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def store_chunks(chunks: list[dict], embeddings: list[list[float]], collection) -> None:
    """Upsert chunks and their embeddings into the vector DB."""
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings,
    )


def ingest(reset: bool = True) -> int:
    """Load, chunk, embed and store the configured DATASET. Returns chunk count."""
    collection = get_collection(reset=reset)
    chunks = chunk_documents(load_documents())
    embeddings = embed_texts([c["text"] for c in chunks])
    store_chunks(chunks, embeddings, collection)
    return len(chunks)


if __name__ == "__main__":
    n = ingest()
    print(f"Stored {n} chunks in '{config.COLLECTION_NAME}' ({config.CHROMA_DIR}/)")
