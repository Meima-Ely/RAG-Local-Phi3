# RAG Local (Offline Analyzer)

Ce projet est une application d'analyse de documents (PDF, DOCX, DOC) fonctionnant **entièrement hors ligne** et localement sur votre machine. Elle utilise l'IA (via Ollama) pour lire, comprendre et répondre à vos questions sur vos documents, garantissant une confidentialité totale.

## Fonctionnalités

- **100% Hors Ligne** : Aucune donnée ne quitte votre ordinateur.
- **Support Multi-Formats** : Ingestion de fichiers PDF, DOCX et DOC.
- **Support OCR** : Peut lire les PDF scannés (si les outils sont installés).
- **Interface Chat** : Posez des questions naturelles sur vos documents.
- **Citations** : L'IA cite les pages où elle a trouvé l'information (ex: `[3]`).
- **Langues** : Détection automatique et réponses en Français, Anglais ou Arabe.

## Prérequis

Avant de lancer l'application, assurez-vous d'avoir installé les éléments suivants :

1.  **Python 3.10+** : [Télécharger Python](https://www.python.org/downloads/)
2.  **Ollama** : [Télécharger Ollama](https://ollama.com/)
    -   Une fois installé, téléchargez les modèles recommandés :
        ```bash
        ollama pull phi3:3.8b-mini-instruct-4k-q4_k_m
        ollama pull nomic-embed-text
        ```
    -   Assurez-vous qu'Ollama tourne en arrière-plan (l'icône doit être visible dans la barre des tâches).

### Optionnel (pour des fonctionnalités avancées)
-   **LibreOffice** : Nécessaire uniquement pour les fichiers `.doc` et `.docx` (les convertit en PDF temporairement).
-   **OCRmyPDF** : Nécessaire si vous voulez que l'IA lise des images scannées de texte.

## Installation

1.  Ouvrez un terminal (PowerShell ou CMD) dans le dossier du projet.
2.  (Optionnel mais recommandé) Créez un environnement virtuel :
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
3.  Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Lancement

Pour démarrer l'application, exécutez la commande suivante dans votre terminal :

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
*Note : Vous pouvez aussi simplement lancer `python app.py`.*

Une fois lancé, ouvrez votre navigateur et allez à l'adresse :
**http://127.0.0.1:8000**

## Utilisation

1.  **Connexion** : L'interface vérifiera automatiquement la connexion à l'API et à Ollama via les indicateurs en haut à droite.
2.  **Upload** : Cliquez sur le trombone ou bouton d'upload pour choisir un fichier.
3.  **Question** : Tapez votre question dans la barre de chat.
    -   L'IA analysera le document, l'ajoutera à sa base de données vectorielle locale, et répondra.
4.  **Reset** : Si vous voulez effacer la mémoire de l'IA, utilisez le bouton "Reset DB".

## Dépannage

-   **Erreur "Chroma DB"** : Si la base de données semble corrompue, l'application tentera de la réparer seule. Si cela persiste, supprimez manuellement le dossier `chroma_db_data`.
-   **Ollama unreachable** : Vérifiez qu'Ollama est bien lancé sur votre machine (`ollama list` dans un terminal pour vérifier).
-   **.doc non supporté** : Installez LibreOffice et assurez-vous que `soffice.exe` est accessible.

## Structure du Projet

-   `app.py` : Le serveur backend (cerveau de l'application).
-   `app.js` : La logique de l'interface utilisateur.
-   `index.html` : L'interface visuelle.
-   `chroma_db_data/` : Dossier contenant la base de données vectorielle (créé automatiquement).
