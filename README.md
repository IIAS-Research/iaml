# IAML

Plateforme d'AutoML développée par l'IIAS. L'objectif est de simplifier la réalisation des projets de DS par la génération et l'execution automatique de PipeLines.
L'aspect Data est traité par le développement package Python. Une interface web sera également développée pour que l'outil puisse être utilisé par tous.

- **Documentation complète** : [http://docs.example.invalid/automl/main](http://docs.example.invalid/automl/main)
- Autres branches : [http://docs.example.invalid/automl/[BRANCH_NAME]](http://docs.example.invalid/automl/[BRANCH_NAME])

***
## 🔧 1 - Installation

### En tant que projet

Pour installer ce projet, suivez les étapes suivantes :

1. Installer Python 3 ([télécharger](https://www.python.org/downloads/))
2. Installer `uv` (le plus simple est de l'installer en global, hors d'un environnement virtuel)
3. Installer les dépendances (la création de l'environnement virtuel est automatique) :
    - Avec support pour notebooks Jupyter et mkDocs, exécuter `uv sync`
    - Autrement, exécuter `uv sync --no-dev`

### En tant que bibliothèque

Insérer une des lignes suivantes dans votre `requirements.txt` :

```bash
iaml @ git+ssh://git@gitlab.example.invalid/DataScience/automl
iaml[cudf] @ git+ssh://git@gitlab.example.invalid/DataScience/automl # Avec cuDF
```

***
## 🚀 2 - Lancement

TODO

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

- MKDOCS version 1.5.3

***
## 💪 5 -  Crédits

- Rudy MERIEUX
