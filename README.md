# **Agent de Contextualisation RAG**

Bienvenue dans votre application **Agent de Contextualisation RAG**, une solution d'intégration et de contextualisation des connaissances basée sur l'IA. Cette application utilise un modèle de récupération de documents (RAG) pour fournir des réponses précises basées uniquement sur les données spécifiées.

---

## **Fonctionnalités**

- **Chargement des fichiers** : Chargez des fichiers CSV ou utilisez les données préchargées pour alimenter la base de connaissances.  
- **Vectorisation des documents** : Utilisation des embeddings via `HuggingFace` pour stocker et rechercher des documents.  
- **Chat contextuel** : Fournit des réponses basées uniquement sur les données récupérées.  
- **Calcul de la pertinence** : Évaluez la similarité entre la réponse générée et les documents source.  
- **Interface utilisateur intuitive** : Une interface Streamlit conviviale pour interagir avec le modèle.

---

## **Prérequis**

Avant de commencer, assurez-vous que les éléments suivants sont installés sur votre machine :

- **Python** : Version 3.10.12
- **Docker** : Pour déployer l'application dans un conteneur (facultatif).

---

## **Installation**

1. Clonez ce dépôt sur votre machine locale :
   ```bash
   git clone https://github.com/Riadh5896/Contextualisation-V2/tree/main

2. Créez un environnement virtuel et activez-le :
   ```bash
     python3.10 -m venv venv
     source venv/bin/activate

3. Clonez ce dépôt sur votre machine locale :
   ```bash
   pip install -r requirements.txt

4. Exécutez l'application :
   ```bash
   streamlit run app.py



