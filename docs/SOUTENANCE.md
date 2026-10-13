# Soutenance — Agent conversationnel RAG sur la documentation financière interne

Formation LLM Engineer · Auteur : Jérémie Coste · Environnement : DGX Spark + JupyterLab

---

## 1. Le projet en 30 secondes
Un assistant de chat pour le département ALM d'un assureur vie : il répond en français aux questions sur 63 DIC de placements financiers, **cite le document et la page**, garde le fil de la conversation et refuse de répondre quand l'information n'est pas dans les documents. Le tout avec un LLM **open weights en local** (aucune donnée ne quitte la machine).

**Preuve :** F1 BERTScore **0,754** (seuil 0,60) sur 619 questions — **0,758** sur les 419 hors sous-échantillon de réglage — avec les réserves du §7.

## 2. Contexte et contraintes
- Pas d'API tierce, modèles open weights ; sources citées ; mode conversationnel ; vecteurs stockés localement (ChromaDB) ; F1 BERTScore > 60 %.
- Cloud Sandbox remplacé par mon **DGX Spark** : `Qwen3.8-27B-NVFP4` (vLLM), `bge-m3`, `bge-reranker-v2-m3`, LangChain, ChromaDB.
- Livrable central : **templates de prompts** (`prompts/`) + code Python + notebooks exécutés.

## 3. Démarche
1. **Inspecter avant de coder** : 63 PDF natifs (pas d'OCR), 250 chunks / 619 questions dans le dataset ; 63 % des questions sont génériques (« ce produit ») ; réponses de référence courtes (médiane 13 mots).
2. **Choisir par la mesure** : 4 embeddings comparés (bge-m3 meilleur), reranker (+10 points de hit@5), taille de chunk, en-tête produit.
3. **Mesurer de bout en bout** dès la première version (F1 0,731) puis **diagnostiquer** : réponses trop longues et refus injustifiés → corrections de prompt mesurées une à une.
4. **Contrôler le score** : réponse constante, score rééchelonné, sous-échantillon de réglage (200) séparé des 419 autres — sachant que le tout premier diagnostic a été fait sur les 619 résultats.

## 4. Architecture
```
question (+ historique) → reformulation autonome → ChromaDB (bge-m3, top-20) → reranker (top-5)
  → LLM local (prompt système + extraits numérotés) → réponse [n] → sources (produit, fichier, page) | refus
```
| Fichier | Rôle |
|---|---|
| `prompts/` | tous les templates (voir `prompts/README.md`) |
| `rag/ingest.py` | parsing, nettoyage, catalogue LLM, découpage avec pages |
| `rag/embeddings.py`, `store.py` | embeddings locaux, ChromaDB, reranker |
| `rag/pipeline.py` | chaînes LangChain, mémoire, citations |
| `rag/evaluation.py`, `sweep.py` | retrieval, BERTScore, comparaison de configurations |

## 5. Choix de conception et justification
- **Sources par construction** : le LLM cite des numéros d'extraits, le code les traduit en (fichier, page) — le LLM n'invente donc pas de références.
- **Page exacte** : texte concaténé avant découpage, page retrouvée par offset.
- **Mémoire par reformulation** : la recherche vectorielle a besoin d'une question autonome ; un suivi comme « et sa société de gestion ? » ne trouverait rien tel quel.
- **En-tête produit dans chaque chunk** : indispensable pour les questions nommées (36 % → 81,5 % de retrieval) ; coûte 2,4 points sur le dataset générique.
- **Reranker** : les passages quasi identiques entre ETF Amundi se concurrencent ; le cross-encoder les départage mieux.
- **Refuser plutôt qu'inventer** : choix de produit pour un usage financier, assumé même s'il coûte du score.

## 6. Difficultés rencontrées et corrections (à citer)
| Constat | Correction |
|---|---|
| Nom du produit introuvable pour 6 DIC | fenêtre autour du premier ISIN dans l'extrait donné au LLM → 63/63 |
| Le LLM énumérait tous les produits (39,5 mots/réponse) | règle « une phrase, extrait le plus pertinent, produit entre parenthèses » |
| 35 % de refus après ce correctif, alors que l'extrait contenait la réponse | refus limité à l'absence totale d'information |
| Métrique de retrieval favorable aux gros chunks | arbitrage sur le score de bout en bout |
| Avertissement de dépréciation LangChain (historique) | classe d'historique minimale |

## 7. Résultats et lecture critique
- **F1 BERTScore** : 0,754 (619) · 0,758 (419 hors réglage) · 0,793 (corpus fermé 250 chunks).
- **Mais** : une réponse constante « je n'ai pas trouvé » obtient déjà **0,637** ; en score rééchelonné 0,345 (pipeline) contre 0,032 (constante). Le seuil de 60 % est franchi, mais il discrimine mal : je m'appuie aussi sur le retrieval (top-5 : 64 % sur l'index PDF) et l'analyse d'erreurs.
- 17 % de refus (58 vrais échecs de retrieval, 47 sur-prudences).
- Détails : `resultats/`, notebook 04.

## 8. Démonstration suggérée (3–4 minutes)
1. Notebook `01` : GPU, modèle servi, 63 DIC, ambiguïté du dataset.
2. Notebook `02` : tableaux de comparaison (embeddings, reranker, en-tête), index ChromaDB.
3. Notebook `03` : question nommée avec sources ; conversation à 4 tours ; refus hors corpus ; injection.
4. Notebook `04` : tableau des scores, contrôle « réponse constante », analyse des refus.

## 9. Limites et améliorations
Voir `docs/RAPPORT.md` : une seule exécution par configuration, réglage sur le dataset d'évaluation, faux refus, questions ambiguës sans clarification, test nommé synthétique, pas de comparaison Mistral/Llama, pas d'API. Pistes : recherche hybride, clarification, tables, LLM-juge de fidélité.

## 10. Questions probables
**Comment garantissez-vous que les sources sont correctes ?** Le LLM ne cite que des numéros d'extraits fournis ; le code produit fichier/page depuis les métadonnées ; l'extrait est conservé pour vérification (notebook 03).

**Pourquoi 75 % alors que le seuil est de 60 % ?** C'est le score brut demandé ; je rapporte aussi qu'une réponse vide obtient 63,7 % : le seuil protège peu, d'où les mesures complémentaires.

**Pourquoi garder l'en-tête alors qu'il baisse le score ?** Le dataset ne contient presque que des questions génériques ; en usage réel les équipes nomment le fonds, et l'en-tête y multiplie le retrieval par 2,3.

**Qu'est-ce qui reste faible ?** Voir §9.

**Un modèle Mistral ou Llama aurait-il été meilleur ?** Non testé ; le serveur local sert Qwen (open weights). L'architecture est indépendante du modèle (variable `LLM_MODEL`).

## 11. Où trouver quoi
`README.md`, `prompts/`, `notebooks/`, `resultats/`, `docs/RAPPORT.md`, `docs/RUNBOOK_JUPYTERLAB.md`.
