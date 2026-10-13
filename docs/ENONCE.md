# Énoncé du projet (résumé)

**Contexte.** Le département ALM d'un grand assureur vie investit sur les marchés financiers pour assurer la rentabilité de ses contrats, en maîtrisant le risque. Chaque placement s'accompagne d'un **document d'informations clés (DIC)**, harmonisé au niveau européen. Le volume d'investissements rend la synthèse de ces documents difficile.

**Mission (LLM Engineer).** Construire un **assistant de chat** permettant aux équipes ALM de retrouver des informations dispersées dans les DIC, selon une architecture **RAG**.

**Contraintes.**
1. Confidentialité : aucun outil tiers ni API, modèles **open weights** (Mistral, Llama, etc.) exécutés dans le SI.
2. Toute réponse trouvée dans un document est **sourcée** (retrouver le document d'origine).
3. Usage **conversationnel** : l'historique des messages est pris en compte.
4. Stockage **local** des vecteurs d'embedding (ChromaDB ou FAISS).

**Étapes.**
1. Découpage des PDF (paramètres de chunking) et embedding adapté à la langue et au contexte, stockage local.
2. Pipeline RAG respectant les contraintes (modèle open weights, sources, mémoire conversationnelle).
3. Évaluation : **F1 BERTScore > 60 %** sur le dataset fourni (`corpus.json`, `queries.json`, `relevant_docs.json`, `answers.json`, `errors.json`, joints par `uuid`) pour que l'assistant puisse être déployé.

**Environnement.** Prévu : Cloud Sandbox (2 GPU 24 Gio, JupyterLab, 6 h). Utilisé ici : **DGX Spark + JupyterLab**, comme pour le projet *assistant service client*.
