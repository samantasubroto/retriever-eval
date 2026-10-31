import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "corpus"
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters
EMBED_BATCH_SIZE = 100

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
GOLDEN_PATH = "data/eval/rag_golden_dataset_15.json"
K_VALUES = (3, 5, 10)
JUDGE_K = 5
