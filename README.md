# POC AutoML
POC : Plateforme d'AutoML pour l'IIAS.

**Complet documentation here** : [Documentation](http://docs.example.invalid/poc_automl/main)
or here for others branches : http://docs.example.invalid/poc_automl/[BRANCH_NAME]

***
## 🔧 1 - Installation

Pour installer ce projet, suivez les étapes suivantes

1. Installer Python 3 ([télécharger](https://www.python.org/downloads/))
2. Exécuter `source ./.venv/bin/python3`
3. Exécuter `pip install -r requirements.txt`


***
## 🚀 2 - Lancement

Le seul objectif de ce projet est de mettre à disposition la documentation des bonnes pratiques. Seule une régénération locale du site web statique peut être lancée :
- Lancer à la racine du projet `python3 -m mkdocs build -d public`

***
## 💡 3 - Informations générales

Structures des fichiers : 

- src/scripts/ : Contients les fichiers de scripts python
- notbooks/ : Contients les notebooks
- libs/ : Contients les librairies python
- data/ : Contients les fichiers base de données (git-ignoré)
- assets/ : Contients les différents fichiers SQL, JSON, XML..
- docs/ : Contients les fichiers markdown permettant la génération de la documentation
- tests : Contients les tests unitaires et d'intégrations 

***
## 📌 4 - Outils et packages utilisés

- MKDOCS version 1.4.2

***
## 💪 5 -  Crédits

- Rudy MERIEUX

