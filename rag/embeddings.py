"""Embeddings locaux (sentence-transformers) compatibles LangChain."""
from langchain_core.embeddings import Embeddings

from .config import EMBED_MODEL

# Préfixes / instructions propres à chaque famille de modèles
PRESETS = {
    "intfloat/multilingual-e5-large": {"query_prefix": "query: ", "doc_prefix": "passage: "},
    "Qwen/Qwen3-Embedding-0.6B": {"query_prompt_name": "query"},
}


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL, device: str | None = None, batch_size: int = 32):
        import torch
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.batch_size = batch_size
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        self.model = SentenceTransformer(model_name, device=device, model_kwargs=kwargs)
        self.model.max_seq_length = min(self.model.max_seq_length or 512, 512)
        self.preset = PRESETS.get(model_name, {})

    def _encode(self, texts, **kw):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False, **kw).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        p = self.preset.get("doc_prefix", "")
        return self._encode([p + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        p = self.preset.get("query_prefix", "")
        kw = {"prompt_name": self.preset["query_prompt_name"]} if "query_prompt_name" in self.preset else {}
        return self._encode([p + text], **kw)[0]
