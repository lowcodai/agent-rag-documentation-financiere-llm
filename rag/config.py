import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DIC_DIR = BASE_DIR / "data" / "dic"
EVAL_DIR = BASE_DIR / "data" / "dataset_eval"
CATALOG_PATH = BASE_DIR / "data" / "catalogue_dic.json"
PROMPTS_DIR = BASE_DIR / "prompts"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "vectorstore"))

# LLM open weights servi localement par vLLM (API compatible OpenAI) : aucune donnée ne quitte le SI
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "")  # vide = premier modèle servi par vLLM

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))        # caractères
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))  # caractères
WITH_HEADER = os.getenv("WITH_HEADER", "1") == "1"   # en-tête « Document : produit (société) » dans chaque chunk
TOP_K = int(os.getenv("TOP_K", 5))
FETCH_K = int(os.getenv("FETCH_K", 20))               # candidats avant reranking
COLLECTION = "dic_pdf"
MAX_HISTORY_TURNS = 6
