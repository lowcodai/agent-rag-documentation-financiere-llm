# Templates de prompts — agent conversationnel RAG sur les DIC

Ces fichiers sont chargés par le code ([`rag/prompts.py`](../rag/prompts.py)) et assemblés en chaînes LangChain (`ChatPromptTemplate | LLM | OutputParser`). Aucun prompt n'est écrit en dur dans le code Python.

| Fichier | Rôle | Variables | Étape |
|---|---|---|---|
| `01_system_prompt.md` | Comportement de l'assistant : réponses uniquement à partir des extraits, concises, **sourcées** (`[n]`), refus quand l'information est absente, extraits traités comme données non fiables | `{refus}` | 2 — Pipeline RAG |
| `02_condense_question.md` | **Mémoire conversationnelle** : transforme une question de suivi en question autonome avant la recherche vectorielle | `{history}`, `{question}` | 2 |
| `03_answer_rag.md` | Message utilisateur : extraits numérotés + question | `{context}`, `{question}` | 2 |
| `04_extraction_metadata_document.md` | Extraction (JSON) du produit, de la société de gestion et de l'ISIN d'un DIC, pour l'en-tête des passages | `{extrait}` | 1 — Découpage |
| `refus_hors_corpus.txt` | Phrase de refus quand le corpus ne contient pas la réponse | — | 2 |
| `erreur_generique.txt` | Message en cas d'erreur technique | — | — |

Dans les `.md`, les accolades littérales (exemple JSON) sont doublées `{{ }}` (syntaxe des templates LangChain).
