"""Base vectorielle locale ChromaDB (persistée sur disque, aucun service externe)."""
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import CHROMA_DIR


def get_store(name: str, embeddings, persist_dir=CHROMA_DIR) -> Chroma:
    return Chroma(collection_name=name, embedding_function=embeddings, persist_directory=str(persist_dir),
                  collection_metadata={"hnsw:space": "cosine"})


def build_store(name: str, docs: list[Document], embeddings, ids: list[str] | None = None,
                persist_dir=CHROMA_DIR, batch: int = 128) -> Chroma:
    """(Re)crée la collection `name` à partir des documents."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    store = get_store(name, embeddings, persist_dir)
    ids = ids or [f"{d.metadata['source']}::{d.metadata['chunk']}" for d in docs]
    for i in range(0, len(docs), batch):
        store.add_documents(docs[i:i + batch], ids=ids[i:i + batch])
    return store


class Reranker:
    """Reranking optionnel par cross-encoder (second étage, plus précis que la similarité d'embeddings)."""

    def __init__(self, model_name: str):
        import torch
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device="cuda" if torch.cuda.is_available() else "cpu",
                                  model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else {})

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if not docs:
            return docs
        scores = self.model.predict([(query, d.page_content) for d in docs], show_progress_bar=False)
        order = sorted(range(len(docs)), key=lambda i: -scores[i])[:top_k]
        return [docs[i] for i in order]
