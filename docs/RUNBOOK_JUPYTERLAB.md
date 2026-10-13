# Runbook — relancer JupyterLab (projet agent RAG sur les DIC)

Contexte : JupyterLab est installé dans `~/projet_rag/.venv` sur le DGX Spark (`ssh dgxspark`). Il n'écoute que sur `127.0.0.1:8898` côté Spark (le port 8899 est réservé au projet *assistant service client*) ; on y accède depuis le Mac par un tunnel SSH. Le LLM (vLLM) doit répondre sur `localhost:8000` côté Spark.

## 1. Vérifier les prérequis (depuis le Mac)
```bash
ssh dgxspark 'curl -s localhost:8000/v1/models | head -c 120; echo; ls ~/projet_rag/.venv/bin/jupyter ~/projet_rag/vectorstore | head'
```
Si le LLM ne répond pas, le démarrer relève du runbook vLLM du dépôt `itshaker-dgx-spark-V2` (pas de ce projet).

## 2. Démarrer JupyterLab sur le Spark (avec un nouveau token)
```bash
TOKEN=$(openssl rand -hex 16); echo "token=$TOKEN"
ssh dgxspark "cd ~/projet_rag && nohup .venv/bin/jupyter lab --no-browser --ip=127.0.0.1 --port=8898 --ServerApp.token=$TOKEN --notebook-dir=\$HOME/projet_rag > jupyter.log 2>&1 < /dev/null &"
```
Notez le token affiché (il n'est pas stocké ailleurs).

## 3. Ouvrir le tunnel SSH (depuis le Mac)
```bash
ssh -f -N -o ExitOnForwardFailure=yes -L 8898:127.0.0.1:8898 dgxspark
```

## 4. Ouvrir dans le navigateur
`http://127.0.0.1:8898/lab/tree/notebooks/04_evaluation.ipynb?token=<TOKEN>`
Noyau à choisir : **Python 3 (rag-dic)**. Ordre d'exécution : `01` → `02` (construit l'index ChromaDB) → `03` → `04` (≈ 25 min : génération des 619 réponses, puis du corpus fermé). Les résultats sont écrits dans `~/projet_rag/resultats/`.

## 5. Exécution sans interface (équivalent)
```bash
ssh dgxspark 'cd ~/projet_rag/notebooks && for n in 01 02 03 04; do ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=rag-dic ${n}_*.ipynb; done'
```

## 6. Diagnostic
| Symptôme | Vérification / action |
|---|---|
| Page inaccessible | Tunnel absent : `lsof -iTCP:8898 -sTCP:LISTEN` ; relancer l'étape 3 |
| « Address already in use » côté Spark | `ssh dgxspark 'pkill -f "jupyter-lab.*8898"'` puis étape 2 |
| Récupérer le token d'un serveur en cours | `ssh dgxspark "pgrep -fa jupyter-lab \| grep -o 'token=[0-9a-f]*'"` |
| Journal du serveur | `ssh dgxspark 'tail -30 ~/projet_rag/jupyter.log'` |
| Noyau « rag-dic » absent | `ssh dgxspark 'cd ~/projet_rag && .venv/bin/python -m ipykernel install --user --name rag-dic --display-name "Python 3 (rag-dic)"'` |
| Erreurs de connexion au LLM | Étape 1 ; variables `LLM_BASE_URL` / `LLM_MODEL` si le port a changé |
| Génération très lente | vLLM partagé avec d'autres travaux : `curl -s localhost:8000/metrics \| grep num_requests` |
| `429` HuggingFace au chargement d'un modèle | Sans conséquence (nouvelle tentative automatique) ; un `HF_TOKEN` supprime l'avertissement |

## 7. Arrêter proprement
```bash
ssh dgxspark 'pkill -f "jupyter-lab.*8898"'
pkill -f "ssh -f -N.*8898"
```
