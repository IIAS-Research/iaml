"""Construire par soustraction dans les familles du registre.

Chaque famille utilise les mêmes opérations que choice. Une suppression sans
correspondance est une erreur ; les exclusions restent locales à la recette."""

from iaml.flow import Int, normalizers, predictors, use
from iaml.steps import (
    KNeighborsClassifier,
    MaxAbsScaler,
    UnitNormScaler,
    RandomForestClassifier,
    SimpleImputer,
)


def build_pipeline():
    # Toutes les normalisations enregistrées, sauf ces deux classes.
    normalization = (
        normalizers()
        .remove(UnitNormScaler, MaxAbsScaler)
        .named("normalize")
    )

    # Toute la famille des prédicteurs, sauf le classificateur à voisins.
    # IAML conserve la vérification d'applicabilité aux données et à la cible :
    # seuls les prédicteurs compatibles peuvent produire des candidats.
    models = (
        predictors()
        .remove(KNeighborsClassifier)
        .named("predictor")
    )

    # find_all(Class) retourne une sélection, même avec zéro ou une correspondance.
    # configure applique les paramètres à toutes les variantes sélectionnées.
    models.find_all(RandomForestClassifier).configure(
        n_estimators=Int(100, 500, initial=200),
    )

    # Les exclusions contraignent aussi les remplacements par l'optimizer :
    # un composant retiré ne revient pas depuis le registre global.
    # La famille est résolue puis figée pour chaque lancement de recherche.
    return use(SimpleImputer).named("cleaning") >> normalization >> models
