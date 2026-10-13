"""Compare des configurations de bout en bout (F1 BERTScore) : python -m rag.sweep 900:150:1 900:150:0 ..."""
import json
import sys
import time

import pandas as pd

from .config import BASE_DIR, EMBED_MODEL, RERANK_MODEL
from .embeddings import LocalEmbeddings
from .evaluation import bertscore, load_eval, run_generation
from .ingest import build_chunks
from .pipeline import RagAssistant
from .store import Reranker, build_store


def main(specs: list[str]):
    ev = load_eval()
    cat = json.loads((BASE_DIR / "data" / "catalogue_dic.json").read_text(encoding="utf-8"))
    emb, rr = LocalEmbeddings(EMBED_MODEL), Reranker(RERANK_MODEL)
    out_dir = BASE_DIR / "resultats" / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = BASE_DIR / "resultats" / "comparaison_configs.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    for spec in specs:
        size, ov, hdr = spec.split(":")
        size, ov, hdr = int(size), int(ov), hdr == "1"
        docs = build_chunks(cat, chunk_size=size, chunk_overlap=ov, with_header=hdr)
        store = build_store(f"sweep_{size}_{int(hdr)}", docs, emb)
        bot = RagAssistant(store, k=5, reranker=rr, fetch_k=20)
        t = time.time()
        df = run_generation(bot, ev)
        df["f1"] = bertscore(df.reponse.tolist(), df.reference.tolist())
        tag = f"chunk{size}_ov{ov}_hdr{int(hdr)}"
        df.to_csv(out_dir / f"{tag}.csv", index=False)
        rows = [r for r in rows if r["config"] != tag] + [{
            "config": tag, "chunk_size": size, "overlap": ov, "en_tete_produit": hdr, "n_chunks": len(docs),
            "f1_bertscore": round(df.f1.mean(), 4), "f1_median": round(df.f1.median(), 4),
            "part_introuvable": round(float((~df.trouve).mean()), 3),
            "mots_par_reponse": round(float(df.reponse.str.split().str.len().mean()), 1),
            "duree_s": round(time.time() - t)}]
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(rows[-1], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
