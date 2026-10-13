from functools import lru_cache
from .config import PROMPTS_DIR


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Charge un template depuis prompts/ (aucun prompt n'est écrit en dur dans le code)."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def refus() -> str:
    return load("refus_hors_corpus.txt")
