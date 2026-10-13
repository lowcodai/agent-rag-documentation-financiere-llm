"""Étape 1 : parsing des PDF, nettoyage, catalogue des documents et découpage en chunks."""
import json
import re
import unicodedata
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CATALOG_PATH, CHUNK_OVERLAP, CHUNK_SIZE, DIC_DIR, WITH_HEADER
from .prompts import load


def clean_text(s: str) -> str:
    """Normalise le texte extrait : ligatures (ﬁ→fi), espaces insécables, césures, lignes vides."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\x00", "").replace("­", "").replace("", "•")
    s = re.sub(r"(\w)-\n(?=[a-zàâçéèêëîïôûùüÿ])", r"\1", s)   # mot coupé en fin de ligne
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" ?\n ?", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Retourne [(numéro de page, texte nettoyé)] ; PyMuPDF respecte l'ordre de lecture des DIC à 2 colonnes."""
    with pymupdf.open(path) as doc:
        pages = [(i + 1, clean_text(p.get_text())) for i, p in enumerate(doc)]
    return [(n, t) for n, t in pages if t]


def list_pdfs(pdf_dir: Path = DIC_DIR) -> list[Path]:
    return sorted(pdf_dir.glob("*.pdf"))


# --- catalogue : produit / société de gestion de chaque DIC (extraits par le LLM local) -----------------
_ISIN = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")


def catalog_excerpt(text: str, head: int = 2500, window: int = 600) -> str:
    """Début du document + fenêtre autour du premier ISIN : dans certains DIC le nom du produit n'apparaît
    pas en tête mais juste avant l'ISIN, plus loin dans la page."""
    m = _ISIN.search(text, head)
    if not m:
        return text[:head + window]
    return text[:head] + "\n[...]\n" + text[max(head, m.start() - window):m.end() + 200]


def build_catalog(llm, pdf_dir: Path = DIC_DIR, path: Path = CATALOG_PATH, force: bool = False) -> dict:
    catalog = json.loads(path.read_text(encoding="utf-8")) if path.exists() and not force else {}
    chain = ChatPromptTemplate.from_template(load("04_extraction_metadata_document.md")) | llm | JsonOutputParser()
    todo = [p for p in list_pdfs(pdf_dir) if p.name not in catalog]
    inputs = [{"extrait": catalog_excerpt("\n".join(t for _, t in load_pdf(p)))} for p in todo]
    results = chain.batch(inputs, config={"max_concurrency": 8}, return_exceptions=True)
    for p, r in zip(todo, results):
        r = r if isinstance(r, dict) else {}
        catalog[p.name] = {"produit": r.get("produit") or p.stem, "societe_gestion": r.get("societe_gestion"),
                           "isin": r.get("isin")}
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    return catalog


def header(meta: dict | None, source: str) -> str:
    meta = meta or {}
    produit = meta.get("produit") or Path(source).stem
    soc = meta.get("societe_gestion")
    return f"Document : {produit}" + (f" ({soc})" if soc else "")


# --- découpage -------------------------------------------------------------------------------------------------
def chunk_pdf(path: Path, catalog: dict | None = None, chunk_size: int = CHUNK_SIZE,
              chunk_overlap: int = CHUNK_OVERLAP, with_header: bool = WITH_HEADER) -> list[Document]:
    """Découpe un DIC en chunks. Le texte de toutes les pages est concaténé avant découpage (les sections
    chevauchent souvent deux pages) ; la page de chaque chunk est retrouvée grâce à son offset."""
    pages = load_pdf(path)
    full, starts = "", []
    for n, t in pages:
        starts.append((len(full), n))
        full += t + "\n\n"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""])
    meta = (catalog or {}).get(path.name)
    head = header(meta, path.name)
    docs = []
    for i, d in enumerate(splitter.create_documents([full.strip()])):
        off = d.metadata["start_index"]
        page = [n for s, n in starts if s <= off][-1]
        text = d.page_content.strip()
        docs.append(Document(
            page_content=(head + "\n" + text) if with_header else text,
            metadata={"source": path.name, "page": page, "chunk": i, "produit": head.removeprefix("Document : ")}))
    return docs


def build_chunks(catalog: dict | None = None, pdf_dir: Path = DIC_DIR, **kw) -> list[Document]:
    return [d for p in list_pdfs(pdf_dir) for d in chunk_pdf(p, catalog, **kw)]
