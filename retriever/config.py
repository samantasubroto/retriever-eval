import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "corpus"
CHUNKING = os.getenv("CHUNKING", "recursive")  # "recursive" or "fixed"
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters
EMBED_BATCH_SIZE = 100

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
GOLDEN_PATH = "data/eval/rag_golden_dataset_15.json"
K_VALUES = (4,)
JUDGE_K = 4

SEARCH_STRATEGY = os.getenv("SEARCH_STRATEGY", "similarity")  # "similarity", "mmr" or "section_cap"
FETCH_K = 20  # candidates fetched before diversifying
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))  # 1.0 = pure relevance, 0.0 = pure diversity
MAX_PER_SECTION = 1
