# Agent conversationnel RAG sur la documentation financière interne

Projet de validation de la formation **LLM Engineer** (BLENT) : un agent conversationnel qui répond en langage naturel à des questions sur un corpus de **documents d'informations clés (DIC / KID PRIIPs)** de produits financiers, en s'appuyant sur une architecture **RAG** (Retrieval-Augmented Generation) construite avec **LangChain**.

> ⚠️ Dépôt en cours de construction : l'énoncé détaillé (étapes, critères d'évaluation) reste à reporter dans [`docs/ENONCE.md`](docs/ENONCE.md) ; les sections ci-dessous seront complétées au fil du projet.

## Données
| Élément | Contenu |
|---|---|
| [`data/dic/`](data/dic) | 62 PDF : DIC / KID PRIIPs de fonds et produits (Allianz, Carmignac, Fidelity, Lazard, ODDO BHF, Pictet, Amundi...) |
| [`data/dataset_eval/`](data/dataset_eval) | Jeu d'évaluation du retrieval : `corpus.json` (250 passages), `queries.json` (619 questions), `relevant_docs.json` (passages pertinents par question), `answers.json` (réponses de référence) |

Exemple de contenu interrogeable : [indicateur de risque d'un DIC](docs/exemple_dic_indicateur_risque.png).

## Structure du dépôt
| Dossier | Contenu |
|---|---|
| [`rag/`](rag) | Code Python : ingestion des PDF, découpage, index vectoriel, retriever, chaîne RAG, évaluation |
| [`prompts/`](prompts) | Templates de prompts (aucun prompt en dur dans le code) |
| [`notebooks/`](notebooks) | Notebooks exécutés avec leurs sorties |
| [`data/`](data) | Documents sources et jeu d'évaluation |
| [`resultats/`](resultats) | Résultats des évaluations (retrieval, réponses) |
| [`docs/`](docs) | Énoncé, rapport, support de soutenance |
| [`evidence/`](evidence) | Preuves d'exécution (environnement, GPU) |

## Exécution
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name rag-dic --display-name "Python 3 (rag-dic)"
cd notebooks && ../.venv/bin/jupyter lab
```

## Licence
MIT — voir [LICENSE](LICENSE). Les DIC sont des documents réglementaires publics publiés par leurs sociétés de gestion respectives.
