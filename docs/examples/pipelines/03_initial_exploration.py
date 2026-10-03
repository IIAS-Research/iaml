"""Comparer plusieurs démarrages dans un même périmètre de recherche.

Un démarrage partiel nécessite un optimizer capable de changer de composant,
comme GeneticOptimizer. start() sans argument rétablit le démarrage complet."""

from iaml.flow import Int, choice, use
from iaml.steps import (
    ExtraTreesClassifier,
    LogisticRegression,
    RandomForestClassifier,
    SimpleImputer,
    StandardScaler,
)


def build_variants():
    models = choice(
        use(LogisticRegression).named("logistic"),
        use(
            RandomForestClassifier,
            n_estimators=Int(100, 500, initial=200),
        ).named("forest"),
        use(ExtraTreesClassifier).named("extra_trees"),
    ).named("predictor")

    preparation = (
        use(SimpleImputer).named("cleaning")
        >> use(StandardScaler).named("normalize")
    )

    # Les trois modèles sont présents dans les structures initiales.
    full = preparation >> models

    # Les recettes clonées sont indépendantes, y compris leurs sous-groupes.
    single_initial = full.clone()
    single_initial.predictor.start("logistic")

    subset_initial = full.clone()
    subset_initial.predictor.start("logistic", "forest")

    # .start(...) règle le démarrage ; .remove(...) réduit le périmètre.
    # Les trois recettes autorisent donc toujours les trois modèles, avec
    # les mêmes bornes pour la forêt. Un optimizer capable de remplacer les
    # composants pourra atteindre les modèles absents du démarrage.
    # Un optimizer qui optimise uniquement les paramètres doit refuser un
    # démarrage partiel, car il ne pourrait pas explorer tout ce périmètre.
    return {
        "complete": full,
        "one_initial_model": single_initial,
        "initial_subset": subset_initial,
    }
