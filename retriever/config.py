import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

DATASET = os.getenv("DATASET", "pdf")  # "pdf" (data/*.pdf) or "multihop" (data/multihop/, see evals/prepare_multihop.py)
DATA_DIR = "data"
MULTIHOP_DIR = "data/multihop"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "corpus" if DATASET == "pdf" else "multihop"
ARTICLE_HEADER = os.getenv("ARTICLE_HEADER", "1") == "1"  # multihop: prefix every chunk with "title | source | date"
CHUNKING = os.getenv("CHUNKING", "recursive")  # "recursive" or "fixed"
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters
EMBED_BATCH_SIZE = 100

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
GOLDEN_PATH = "data/eval/rag_golden_dataset_15.json" if DATASET == "pdf" else f"{MULTIHOP_DIR}/golden.json"
K_VALUES = tuple(int(k) for k in os.getenv("K_VALUES", "4").split(","))
JUDGE_K = int(os.getenv("JUDGE_K", "4"))

SEARCH_STRATEGY = os.getenv("SEARCH_STRATEGY", "similarity")  # "similarity", "mmr", "doc_cap" or "hybrid_rerank"
FETCH_K = 20  # candidates fetched before diversifying
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))  # 1.0 = pure relevance, 0.0 = pure diversity
MAX_PER_DOC = 1

# hybrid_rerank: dense + BM25 fused with reciprocal rank fusion (RRF), then reranked
POOL_K = 20  # candidates kept after fusion
FINAL_K = int(os.getenv("FINAL_K", "5"))  # chunks kept after reranking
CANDIDATES_PER_RETRIEVER = 30  # taken from each of dense and BM25 before fusing
RRF_C = 60  # standard RRF constant
RERANK_MODEL = os.getenv("RERANK_MODEL", JUDGE_MODEL)
