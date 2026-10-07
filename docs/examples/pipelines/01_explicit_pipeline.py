"""Construire un pipeline explicite avec des briques réutilisables.

IAML explore les alternatives déclarées et optimise leurs paramètres.
Les fonctions reçoivent les données ; importer ce module ne lance aucun calcul."""

from iaml import IAML
from iaml.flow import Int, choice, optional, use
from iaml.steps import (
    LogisticRegression,
    RandomForestClassifier,
    RobustScaler,
    SimpleImputer,
    SMOTE,
    StandardScaler,
)


def build_pipeline():
    normalization = choice(
        StandardScaler,
        RobustScaler,
    ).named("normalize")

    models = choice(
        use(LogisticRegression).named("logistic"),
        use(
            RandomForestClassifier,
            n_estimators=Int(100, 500, initial=200),
            max_depth=Int(3, 30, initial=15),
        ).named("forest"),
    ).named("predictor")

    # Une classe seule dans choice(...) équivaut à use(Class).
    # .named(...) donne un alias si l'on souhaite cibler une alternative.
    # use(...) permet de donner une configuration ou de démarrer une séquence.
    pipeline = (
        use(SimpleImputer).named("cleaning")
        >> normalization
        >> optional(use(SMOTE)).named("imbalance")
        >> models
    )

    # Les choix explorent toutes leurs alternatives au démarrage par défaut.
    # Ici : 2 normalisations × avec/sans SMOTE × 2 modèles = 8 structures
    # possibles, avant les vérifications d'applicabilité aux données.
    # Les hyperparamètres restent à optimiser dans les domaines déclarés.
    return pipeline


def train(X, y):
    """Entraîner avec la recette explicite, avec les alternatives déclarées."""
    search = IAML(pipeline=build_pipeline(), max_duration=60)
    return search, search.fit(X, y)
