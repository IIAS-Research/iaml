"""Choisir entre des stratégies complètes de préparation et de modèle.

Chaque alternative conserve ses propres transformations et paramètres.
Ces sous-pipelines participent tous à l'exploration initiale."""

from iaml import IAML
from iaml.flow import Float, Int, choice, use
from iaml.steps import (
    LogisticRegression,
    RandomForestClassifier,
    SimpleImputer,
    StandardScaler,
)


def build_pipeline():
    """Partager le nettoyage, puis choisir entre deux stratégies complètes."""
    linear = (
        use(StandardScaler)
        >> use(LogisticRegression, tol=Float(1e-5, 1e-2, initial=1e-4))
    ).named("linear")

    forest = use(
        RandomForestClassifier,
        n_estimators=Int(100, 500, initial=200),
        max_depth=Int(3, 30, initial=15),
    ).named("forest")

    strategy = choice(linear, forest).named("strategy")
    return use(SimpleImputer).named("cleaning") >> strategy


def customize_pipeline():
    """Configurer une stratégie depuis la recette composée."""
    pipeline = build_pipeline()
    pipeline["forest"].configure(n_estimators=300)
    # La forêt garde le domaine [100, 500]. Cette édition ne modifie pas
    # la préparation ni le domaine tol de la stratégie linear.
    return pipeline


def train(X, y):
    """Entraîner les deux stratégies déclarées."""
    search = IAML(pipeline=customize_pipeline(), max_duration=60)
    return search, search.fit(X, y)


# Contrat des stratégies :
# - les deux sous-pipelines sont générés au démarrage, sous réserve de leur
#   applicabilité et du budget ; aucune recherche exhaustive des paramètres
#   ni évaluation terminée de chaque branche n'est garantie ;
# - leurs paramètres sont optimisables dans les domaines propres à chaque
#   branche, avec les valeurs initiales déclarées ;
# - l'optimizer ne remplace pas leur modèle par une classe quelconque du
#   registre global : linear conserve son modèle et forest conserve le sien ;
# - changer atomiquement de sous-pipeline pendant une exploration partielle
#   n'est pas pris en charge ; strategy.start(...) ne peut pas restreindre
#   le démarrage de ce groupe, contrairement aux choix de briques simples.
