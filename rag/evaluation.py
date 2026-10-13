"""Étape 3 : évaluation sur le dataset fourni (retrieval + F1 BERTScore des réponses)."""
import json
import random
import re
import unicodedata

import pandas as pd
from langchain_core.documents import Document

from .config import EVAL_DIR


def load_eval(eval_dir=EVAL_DIR) -> dict:
    names = ["corpus", "queries", "relevant_docs", "answers", "errors"]
    return {n: json.loads((eval_dir / f"{n}.json").read_text(encoding="utf-8")) for n in names}


def corpus_documents(corpus: dict) -> tuple[list[Document], list[str]]:
    """Les 250 chunks du dataset, indexés tels quels (l'identifiant du chunk = uuid du corpus)."""
    ids = list(corpus)
    docs = [Document(page_content=corpus[i], metadata={"source": f"corpus:{i[:8]}", "page": 0, "chunk": k})
            for k, i in enumerate(ids)]
    return docs, ids


# --- retrieval ------------------------------------------------------------------------------------------------
def exact_retrieval(store, ids: list[str], queries: dict, relevant: dict, ks=(1, 3, 5, 10)) -> dict:
    """Hit@k et MRR quand l'index contient exactement les chunks du dataset (`ids[chunk]` = uuid du corpus)."""
    kmax, hits, rr = max(ks), {k: 0 for k in ks}, 0.0
    for uid, q in queries.items():
        got = [ids[d.metadata["chunk"]] for d in store.similarity_search(q, k=kmax)]
        rank = next((r for r, i in enumerate(got, 1) if i in relevant[uid]), None)
        for k in ks:
            hits[k] += rank is not None and rank <= k
        rr += 1 / rank if rank else 0
    n = len(queries)
    return {**{f"hit@{k}": hits[k] / n for k in ks}, "mrr": rr / n}


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", s).lower().replace("’", "'"))


def _grams(s: str, n: int = 4) -> set:
    t = _tokens(s)
    return set(zip(*[t[i:] for i in range(n)])) if len(t) >= n else {tuple(t)}


def coverage(reference_text: str, retrieved_texts: list[str]) -> float:
    """Part des 4-grammes du chunk de référence retrouvés dans les chunks récupérés."""
    ref = _grams(reference_text)
    got = set().union(*[_grams(t) for t in retrieved_texts]) if retrieved_texts else set()
    return len(ref & got) / len(ref)


def pdf_retrieval(retrieve_fn, queries: dict, relevant: dict, corpus: dict, all_chunk_texts: list[str],
                  ks=(1, 3, 5), threshold: float = 0.5) -> dict:
    """Retrieval sur l'index construit depuis les PDF (découpage différent de celui du dataset) : un passage
    de référence est « retrouvé » si ≥ `threshold` de son contenu figure dans les k chunks récupérés.
    Les passages du dataset absents de notre extraction PDF (texte dégradé côté dataset) sont exclus."""
    everything = set().union(*[_grams(t) for t in all_chunk_texts])
    evaluable = {u for u in queries if len(_grams(corpus[relevant[u][0]]) & everything)
                 / len(_grams(corpus[relevant[u][0]])) >= 0.6}
    hits = {k: 0 for k in ks}
    for uid in evaluable:
        texts = [d.page_content for d in retrieve_fn(queries[uid])]
        ref = corpus[relevant[uid][0]]
        for k in ks:
            hits[k] += coverage(ref, texts[:k]) >= threshold
    n = len(evaluable)
    return {**{f"hit@{k}": hits[k] / n for k in ks}, "n_evaluables": n, "n_total": len(queries)}


# --- génération + BERTScore ----------------------------------------------------------------------------------
def bertscore(candidates: list[str], references: list[str], rescale: bool = False, device: str | None = None) -> list[float]:
    """F1 BERTScore par paire (modèle multilingue par défaut de bert_score pour lang='fr')."""
    import torch
    from bert_score import score
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cands = [c if c.strip() else "." for c in candidates]
    _, _, f1 = score(cands, references, lang="fr", device=device, batch_size=32,
                     rescale_with_baseline=rescale, verbose=False)
    return f1.tolist()


def run_generation(assistant, ev: dict, limit: int | None = None) -> pd.DataFrame:
    uids = list(ev["queries"])
    if limit:   # échantillon aléatoire reproductible (le fichier est trié par document)
        uids = random.Random(0).sample(uids, limit)
    res = assistant.answer_batch([ev["queries"][u] for u in uids])
    return pd.DataFrame([{
        "uuid": u, "question": r.question, "reference": ev["answers"][u], "reponse": r.answer, "trouve": r.found,
        "sources": "; ".join(f"{s['source']} p.{s['page']}" for s in r.sources)} for u, r in zip(uids, res)])


# --- questions nommées (test de robustesse de l'en-tête « produit ») -----------------------------------------
def locate_sources(ev: dict, pdf_texts: dict[str, str], min_cov: float = 0.6) -> dict:
    """Pour chaque question, PDF d'origine du passage de référence (meilleure couverture de 4-grammes)."""
    grams = {f: _grams(t) for f, t in pdf_texts.items()}
    out = {}
    for uid, rel in ev["relevant_docs"].items():
        ref = _grams(ev["corpus"][rel[0]])
        f, c = max(((f, len(ref & g) / len(ref)) for f, g in grams.items()), key=lambda x: x[1])
        if c >= min_cov:
            out[uid] = f
    return out


def named_query_retrieval(retrieve_fn, ev: dict, located: dict, catalog: dict, ks=(1, 3, 5)) -> dict:
    """Question enrichie du nom du produit (« … pour le produit X ») : succès si le bon passage du BON
    document figure dans le top-k. Mesure ce que le dataset (questions génériques) ne peut pas mesurer."""
    hits = {k: 0 for k in ks}
    for uid, f in located.items():
        q = ev["queries"][uid].rstrip(" ?") + f" pour le produit {catalog[f]['produit']} ?"
        got = retrieve_fn(q)
        ref = ev["corpus"][ev["relevant_docs"][uid][0]]
        for k in ks:
            texts = [d.page_content for d in got[:k] if d.metadata["source"] == f]
            hits[k] += bool(texts) and coverage(ref, texts) >= 0.5
    n = len(located)
    return {**{f"hit@{k}": hits[k] / n for k in ks}, "n": n}
