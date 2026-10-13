# Agent conversationnel RAG sur la documentation financière interne

Projet de validation de la formation **LLM Engineer** (BLENT) : un assistant de chat pour le département **ALM** (gestion actif/passif) d'un assureur vie, qui retrouve en langage naturel des informations dispersées dans **63 documents d'informations clés (DIC / DICI / KID PRIIPs)**. Architecture **RAG** avec **LangChain**, **ChromaDB** et un LLM **open weights exécuté localement** (`Qwen3.8-27B-NVFP4` servi par vLLM sur un **DGX Spark**) : aucune API tierce, aucune donnée ne quitte la machine.

## Contraintes du projet et réponse apportée
| Contrainte | Réalisation |
|---|---|
| Pas de Cloud public, modèles open weights en local | LLM `Qwen3.8-27B-NVFP4` (vLLM), embeddings `BAAI/bge-m3`, reranker `bge-reranker-v2-m3`, tous locaux |
| Réponses **sourcées** | Citation `[n]` par le LLM → fichier, **page** et produit affichés ; l'extrait source reste consultable |
| Mode **conversationnel** avec historique | Question de suivi reformulée en question autonome avant la recherche ; historique par session |
| Stockage local des vecteurs | **ChromaDB** persistant |
| **F1 BERTScore > 60 %** | **75,4 %** sur les 619 questions (75,8 % sur les 419 hors sous-échantillon de réglage) — lire [les réserves](#résultats) |

## Les 3 étapes du projet
| Étape | Réalisation | Où |
|---|---|---|
| **1. Découpage des PDF et embedding** | PyMuPDF, nettoyage (ligatures, césures), catalogue produit/société par le LLM, chunks 900/150 avec en-tête produit et numéro de page, embeddings bge-m3 comparés à 3 autres modèles, ChromaDB | [`rag/ingest.py`](rag/ingest.py), [`embeddings.py`](rag/embeddings.py), [`store.py`](rag/store.py) · [notebook 01](notebooks/01_environnement_donnees.ipynb), [02](notebooks/02_decoupage_et_embedding.ipynb) |
| **2. Pipeline RAG** | Chaîne LangChain `prompt \| LLM \| parser` : reformulation → recherche top-20 → reranking top-5 → réponse sourcée ; refus quand l'information est absente | [`rag/pipeline.py`](rag/pipeline.py), [`prompts/`](prompts) · [notebook 03](notebooks/03_pipeline_rag_conversationnel.ipynb) |
| **3. Évaluation** | F1 BERTScore sur les 619 questions, contrôles (réponse constante, score rééchelonné), retrieval, analyse des erreurs, variante corpus fermé | [`rag/evaluation.py`](rag/evaluation.py), [`sweep.py`](rag/sweep.py) · [notebook 04](notebooks/04_evaluation.ipynb) |

## Architecture
```
question (+ historique de la session)
  → reformulation en question autonome (si historique)        prompts/02_condense_question.md
  → recherche vectorielle ChromaDB (bge-m3, 20 candidats)
  → reranking cross-encoder (bge-reranker-v2-m3, top-5)
  → LLM local : 01_system_prompt.md + 03_answer_rag.md        réponse concise, citations [n]
  → réponse + « Sources : produit — fichier, page » / refus « information introuvable »
```

## Structure du dépôt
| Dossier | Contenu |
|---|---|
| [`prompts/`](prompts) | **Templates de prompts** (livrable central), décrits dans [`prompts/README.md`](prompts/README.md) |
| [`rag/`](rag) | Code Python : ingestion, embeddings, base vectorielle, pipeline conversationnel, évaluation, chat CLI |
| [`notebooks/`](notebooks) | 4 notebooks **exécutés avec leurs sorties** (`.ipynb`) et exports `.html` |
| [`data/`](data) | 63 DIC (`dic/`), dataset d'évaluation (`dataset_eval/`), catalogue produit/société (`catalogue_dic.json`) |
| [`resultats/`](resultats) | Scores, comparaisons (embeddings, reranker, découpage, configurations), réponses des 619 questions |
| [`docs/`](docs) | [Énoncé](docs/ENONCE.md), [rapport](docs/RAPPORT.md), [support de soutenance](docs/SOUTENANCE.md), [runbook JupyterLab](docs/RUNBOOK_JUPYTERLAB.md) |
| [`evidence/`](evidence) | `nvidia-smi` et modèle servi (preuve de l'environnement) |

## Résultats
Exécutés sur le DGX Spark (notebook 04) :

| Mesure | Valeur |
|---|---|
| **F1 BERTScore**, 619 questions, index construit depuis les PDF | **0,754** (IC95 % [0,745 ; 0,764]) |
| … sur les 419 questions hors sous-échantillon de réglage | **0,758** |
| … variante « corpus fermé » (index = les 250 chunks du dataset) | 0,793 |
| Retrieval (index PDF, reranker) : passage de référence retrouvé dans le top-1 / 3 / 5 | 41 % / 58 % / 64 % |
| Retrieval, questions **nommées** (« … pour le produit X »), avec / sans en-tête produit, top-5 | 81,5 % / 36,0 % |
| Part de refus (« introuvable ») | 17 % |

**Réserves à connaître** (détaillées dans [docs/RAPPORT.md](docs/RAPPORT.md)) :
- Le seuil de 60 % est **peu sélectif** : une réponse constante sans aucune information (le message de refus) obtient déjà **0,637** en score brut. En score *rééchelonné*, le pipeline fait 0,345 contre 0,032 pour la constante.
- 63 % des questions du dataset ne nomment aucun produit (« l'indicateur de risque de ce produit ? ») alors que 63 DIC répondent différemment : le score est plafonné par cette ambiguïté.
- Prompt et découpage ont été réglés en observant le dataset d'évaluation (premier diagnostic sur les 619 questions, corrections suivantes mesurées sur un sous-échantillon fixe de 200) : le score des 419 autres est le plus proche d'une estimation indépendante, sans l'être totalement ; l'en-tête produit coûte ≈ 2,4 points de F1 sur ce dataset (0,779 sans) mais multiplie par 2,3 le retrieval sur questions nommées, cas d'usage réel : choix assumé.
- Réponses du LLM non déterministes ; un faux refus subsiste quand l'information est présente (47 des 105 refus).

## Exécution
Prérequis : un LLM compatible OpenAI sur `http://localhost:8000/v1` (variables `LLM_BASE_URL` / `LLM_MODEL`), un GPU, Python 3.12.
```bash
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130   # adapter à votre CUDA
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name rag-dic --display-name "Python 3 (rag-dic)"
cd notebooks && ../.venv/bin/jupyter lab        # exécuter 01 → 02 (construit l'index ChromaDB) → 03 → 04
```
Chat en ligne de commande (après le notebook 02) : `.venv/bin/python -m rag.chat`. En Python :
```python
from rag.pipeline import default_assistant
bot = default_assistant()
r = bot.ask("Quel est l'indicateur de risque du fonds Pictet - Water ?", session_id="alm")
print(r.render())                                             # réponse + sources (fichier, page)
print(bot.ask("Et sa société de gestion ?", session_id="alm").render())   # suivi dans la même conversation
```
Les résultats publiés ont été produits sur le DGX Spark ; ailleurs, adaptez le LLM (les réponses peuvent varier). Le dossier `vectorstore/` (index Chroma) n'est pas versionné : il se reconstruit avec le notebook 02.

## Données
- **DIC** : 63 documents réglementaires publics (175 pages) de sociétés de gestion (Amundi, SG 29 Haussmann, Pictet, Fidelity, Carmignac, Allianz, Lazard, ODDO BHF, etc.).
- **Dataset d'évaluation** fourni par la formation : 250 chunks, 619 questions, une réponse de référence par question.

## Licence
MIT — voir [LICENSE](LICENSE). Les DIC sont des documents publics publiés par leurs sociétés de gestion respectives et restent la propriété de leurs auteurs.
