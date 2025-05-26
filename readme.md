# Assistant RAG - Chat basé sur un PDF

Ce projet est une application d'assistant conversationnel qui utilise une approche **RAG** pour répondre à des questions sur un document PDF (ici une recette de cookies).  
L'application combine :

- Une API backend avec **FastAPI** qui traite les questions en interrogeant un modèle de langage (LLM) via LangChain et un système de vectorisation.
- Une interface utilisateur en **Streamlit** pour dialoguer en mode chat avec l’assistant.

---

## Fonctionnalités

- Chargement et découpage d’un PDF en morceaux (chunks) pour meilleure recherche.
- Vectorisation des chunks avec embeddings pour la recherche de contexte pertinent.
- Utilisation d’un modèle LLM (Llama3.1 via Ollama) pour générer des réponses contextualisées.
- Interface de chat dynamique avec historique des messages.
- Affichage des étapes de chargement et de préparation des données en temps réel.

---

## Structure du projet

- `rag_utils.py` : Construction de la chaîne RAG, chargement PDF, découpage, embedding, création du retriever.
- `main2.py` : Fonction pour répondre aux questions en tenant compte de l’historique de chat.
- `api.py` : API FastAPI exposant un endpoint `/ask` pour poser des questions.
- `interface.py` : Interface Streamlit qui communique avec l’API et affiche la conversation.
- `Test Input/recette-cookies.pdf` : Document PDF de recette utilisé pour la démonstration.

---

## Installation

1. Cloner le projet :
2. Utiliser pyenv pour utiliser la bonne version de python 3.12.9

```bash
pyenv intall 3.12.9
pyenv local 3.12.9
```

3. Installer toutes les bibliothèques avec poetry

```bash
poetry install
```


## Lancement de l'application
### Lancer le serveur FastAPI 

```bash
poetry run uvicorn api:app --reload
```


### Lancer l’interface Streamlit dans un deuxième terminal

```bash
poetry run streamlit run interface.py
```

