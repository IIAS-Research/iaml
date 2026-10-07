"""Personnaliser les stratégies visibles du preset AutoML.

PipelineSpec.default() contient main et minimal. L'arbre se parcourt et
s'édite avec les mêmes objets que ceux utilisés pour sa construction."""

from iaml.flow import Int, PipelineSpec, use
from iaml.steps import (
    KNeighborsClassifier,
    MaxAbsScaler,
    UnitNormScaler,
    RandomForestClassifier,
    StandardScaler,
)


def build_pipeline():
    """Adapter quelques groupes sans réécrire tout le pipeline AutoML."""
    pipeline = PipelineSpec.default()
    main = pipeline.main

    # Le groupe du défaut propose une variante par classe. On nomme celle avec
    # laquelle commencer ; start sélectionne une identité, comme l'accès par [].
    for scaler in main.normalize.find_all(StandardScaler):
        scaler.named("standard")

    # La méthode remove agit ici sur les alternatives du groupe normalize.
    # Les autres normalisations du défaut restent autorisées.
    main.normalize.remove(UnitNormScaler, MaxAbsScaler).start("standard")

    # Ajouter une configuration nommée au groupe de modèles existant.
    # Une classe peut apparaître avec plusieurs configurations : l'alias permet
    # alors de viser exactement celle-ci, au lieu de sélectionner par classe.
    main.predictor.remove(KNeighborsClassifier).add(
        use(
            RandomForestClassifier,
            n_estimators=Int(50, 150, initial=100),
            max_depth=Int(3, 12, initial=6),
        ).named("forest_small"),
    )

    # Au niveau du pipeline, remove retire un bloc entier.
    main.remove("imbalance")

    # La stratégie minimaliste est visible et éditable séparément :
    # print(pipeline.minimal.minimal_predictor.describe())
    # pipeline.remove("minimal") retire cette stratégie entière.

    # L'alias est unique dans la recette : aucun chemin n'est nécessaire.
    pipeline["forest_small"].configure(n_estimators=120)
    # La valeur simple modifie le départ ; le domaine [50, 150] reste en place.

    return pipeline


def add_business_feature(pipeline, feature_class):
    """Insérer sa propre brique, en conservant le nettoyage du défaut.

    feature_class désigne une classe de Step compatible avec IAML, par exemple
    un calcul de BMI fourni par le projet utilisateur.
    """
    pipeline.add(use(feature_class).named("bmi"), before="cleaning")
    return pipeline


# Utilisation :
# from iaml import IAML
# pipeline = build_pipeline()
# pipeline = add_business_feature(pipeline, BMI)
# search = IAML(pipeline=pipeline)
# search.fit(X, y)
