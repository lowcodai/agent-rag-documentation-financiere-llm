"""Étape 2 : pipeline RAG conversationnel (LangChain) : question de suivi → question autonome → recherche
vectorielle → réponse sourcée générée par un LLM open weights local."""
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import MAX_HISTORY_TURNS, TOP_K
from .llm import get_llm
from .prompts import load, refus

_CITATION = re.compile(r"\s*\[(\d+(?:\s*[,;]\s*\d+)*)\]")


@dataclass
class RagAnswer:
    question: str
    standalone_question: str
    answer: str                       # réponse sans marqueurs de citation
    sources: list[dict] = field(default_factory=list)   # documents réellement cités
    contexts: list[Document] = field(default_factory=list)  # tous les extraits fournis au LLM
    found: bool = True

    def render(self) -> str:
        """Réponse + bloc « Sources » à afficher à l'utilisateur."""
        if not self.sources:
            return self.answer
        lines = [f"  [{s['num']}] {s['produit']} — {s['source']}, page {s['page']}" for s in self.sources]
        return self.answer + "\n\nSources :\n" + "\n".join(lines)


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}" for i, d in enumerate(docs, 1))


def parse_citations(text: str, n_docs: int) -> tuple[str, list[int]]:
    """Extrait les numéros d'extraits cités [1][3] et retourne le texte sans marqueurs."""
    nums = []
    for m in _CITATION.finditer(text):
        nums += [int(x) for x in re.findall(r"\d+", m.group(1))]
    clean = _CITATION.sub("", text).strip()
    return clean, [n for n in dict.fromkeys(nums) if 1 <= n <= n_docs]


class ChatHistory:
    """Historique d'une conversation (en mémoire, une instance par session)."""

    def __init__(self):
        self.messages: list[BaseMessage] = []

    def add_user_message(self, text: str):
        self.messages.append(HumanMessage(text))

    def add_ai_message(self, text: str):
        self.messages.append(AIMessage(text))


class RagAssistant:
    def __init__(self, store, llm=None, k: int = TOP_K, reranker=None, fetch_k: int | None = None):
        self.store, self.k, self.reranker = store, k, reranker
        self.fetch_k = fetch_k or (k * 4 if reranker else k)
        self.llm = llm or get_llm()
        self.histories: dict[str, ChatHistory] = {}
        self._condense = (ChatPromptTemplate.from_template(load("02_condense_question.md"))
                          | self.llm | StrOutputParser())
        self._answer = (ChatPromptTemplate.from_messages([
            ("system", load("01_system_prompt.md")),
            MessagesPlaceholder("history"),
            ("human", load("03_answer_rag.md"))]) | self.llm | StrOutputParser())

    # -- recherche ----------------------------------------------------------------------------------------
    def retrieve(self, query: str) -> list[Document]:
        docs = self.store.similarity_search(query, k=self.fetch_k)
        return self.reranker.rerank(query, docs, self.k) if self.reranker else docs[:self.k]

    # -- mémoire conversationnelle --------------------------------------------------------------------------
    def history(self, session_id: str) -> ChatHistory:
        return self.histories.setdefault(session_id, ChatHistory())

    def reset(self, session_id: str = "default"):
        self.histories.pop(session_id, None)

    @staticmethod
    def _history_text(messages) -> str:
        return "\n".join(f"{'Utilisateur' if m.type == 'human' else 'Assistant'} : {m.content}" for m in messages)

    def _window(self, session_id: str):
        return self.history(session_id).messages[-2 * MAX_HISTORY_TURNS:]

    # -- génération ------------------------------------------------------------------------------------------
    def _finish(self, question: str, standalone: str, docs: list[Document], raw: str) -> RagAnswer:
        text, nums = parse_citations(raw, len(docs))
        found = not text.lower().startswith(refus()[:25].lower())
        if not found:
            return RagAnswer(question, standalone, refus(), [], docs, False)
        nums = nums or [1]    # réponse sans citation : on rattache au premier extrait
        sources = [{"num": n, "source": docs[n - 1].metadata["source"], "page": docs[n - 1].metadata["page"],
                    "produit": docs[n - 1].metadata.get("produit", ""), "extrait": docs[n - 1].page_content}
                   for n in nums]
        return RagAnswer(question, standalone, text, sources, docs, True)

    def ask(self, question: str, session_id: str = "default") -> RagAnswer:
        """Pose une question dans une conversation (l'historique de `session_id` est pris en compte)."""
        past = self._window(session_id)
        standalone = (self._condense.invoke({"history": self._history_text(past), "question": question}).strip()
                      if past else question)
        docs = self.retrieve(standalone)
        raw = self._answer.invoke({"history": past, "refus": refus(),
                                   "context": format_context(docs), "question": question})
        res = self._finish(question, standalone, docs, raw)
        h = self.history(session_id)
        h.add_user_message(question)
        h.add_ai_message(res.answer)
        return res

    def answer_batch(self, questions: list[str], concurrency: int = 16) -> list[RagAnswer]:
        """Questions indépendantes (sans historique), générations parallèles : utilisé pour l'évaluation."""
        docs = [self.retrieve(q) for q in questions]
        raws = self._answer.batch(
            [{"history": [], "refus": refus(), "context": format_context(d), "question": q}
             for q, d in zip(questions, docs)],
            config={"max_concurrency": concurrency}, return_exceptions=True)
        out = []
        for q, d, r in zip(questions, docs, raws):
            out.append(RagAnswer(q, q, "", [], d, False) if isinstance(r, Exception) else self._finish(q, q, d, r))
        return out


def default_assistant(embeddings=None, reranker=None, llm=None) -> "RagAssistant":
    """Assistant branché sur l'index final (collection `dic_pdf`) avec les paramètres de config.py."""
    from .config import COLLECTION, EMBED_MODEL, FETCH_K, RERANK_MODEL, USE_RERANKER
    from .embeddings import LocalEmbeddings
    from .store import Reranker, get_store
    store = get_store(COLLECTION, embeddings or LocalEmbeddings(EMBED_MODEL))
    if reranker is None and USE_RERANKER:
        reranker = Reranker(RERANK_MODEL)
    return RagAssistant(store, llm=llm, reranker=reranker, fetch_k=FETCH_K)
