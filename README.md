# IAML


AutoML platform developed by IIAS. The aim is to simplify the implementation of Data Scientist projects through the automatic generation and execution of pipelines.
The data aspect is handled by the Python package development. A web interface will also be developed so that the tool can be used by everyone.

- **Documentation** : __coming_soon__

***
## 🔧 1 - Installation

### As a project

To install this project, follow these steps:

1. Install Python 3 ([download](https://www.python.org/downloads/))
2. Install `uv` (the easiest way is to install it globally, outside a virtual environment)
3. Install dependencies (virtual environment creation is automatic):
    - With support for Jupyter and mkDocs notebooks, run `uv sync`
    - Otherwise, run `uv sync --no-dev`

### As a library

Insert one of the following lines in your `requirements.txt` file :

```bash
iaml @ git+ssh://git@github.com:iias_research/iaml
iaml[cudf] @ git+ssh://git@github.com:iias_research/iaml # Avec cuDF
```

***
## 🚀 2 - How to run

TODO

***
## 💡 3 - General information

Directory structure :

- src/scripts/ : Contains python script files
- notbooks/ : Contains notebooks
- libs/ : Contains python libraries
- data/ : Contains database files (git-ignored)
- assets/ : Contains various SQL, JSON, XML files..
- docs/ : Contains markdown files for generating documentation
- tests : Contains unit and integration testing

***
## 📌 4 - Tools and packages used

- MKDOCS version 1.5.3

***
## 💪 5 -  Credits

- Rudy MERIEUX
- Robin BOURACHOT
- Hugo RUELLET
