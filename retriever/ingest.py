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


def chunk_pages(pages: list[dict], source: str) -> list[dict]:
    """Chunk each page, keeping source/page metadata and a stable id."""
    chunks = []
    for page in pages:
        for j, text in enumerate(chunk_text(page["text"])):
            chunks.append(
                {
                    "id": f"{source}:p{page['page']}:c{j}",
                    "text": text,
                    "metadata": {
                        "source": source,
                        "page": page["page"],
                        "section_id": f"RAG-{page['page']:03d}",
                        "chunk": j,
                    },
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


def ingest(data_dir: str = config.DATA_DIR, reset: bool = True) -> int:
    """Load every PDF in data_dir, chunk, embed and store. Returns chunk count."""
    collection = get_collection(reset=reset)
    chunks = []
    for pdf in sorted(Path(data_dir).glob("*.pdf")):
        chunks.extend(chunk_pages(load_pdf(pdf), source=pdf.name))
    embeddings = embed_texts([c["text"] for c in chunks])
    store_chunks(chunks, embeddings, collection)
    return len(chunks)


if __name__ == "__main__":
    n = ingest()
    print(f"Stored {n} chunks in '{config.COLLECTION_NAME}' ({config.CHROMA_DIR}/)")
