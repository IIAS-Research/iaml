"""Réutiliser des fragments et créer des variantes indépendantes.

Les constructeurs, add, clone et la composition par >> copient les recettes.
Chaque entraînement compile de nouvelles instances de Step."""

from iaml.flow import Int, choice, normalizers, optional, use
from iaml.steps import (
    LogisticRegression,
    MaxAbsScaler,
    UnitNormScaler,
    RandomForestClassifier,
    RobustScaler,
    SimpleImputer,
    SMOTE,
    StandardScaler,
)


def preparation():
    """Retourner une nouvelle brique de préparation à chaque appel."""
    normalization = normalizers().remove(UnitNormScaler).named("normalize")
    # Le catalogue initial contient une variante par classe. Un alias explicite
    # permet de cibler StandardScaler dans la politique de démarrage.
    for scaler in normalization.find_all(StandardScaler):
        scaler.named("standard")

    return (
        use(SimpleImputer).named("cleaning")
        >> normalization
    )


def models():
    """Décrire deux modèles et le domaine de recherche de la forêt."""
    return choice(
        use(LogisticRegression).named("logistic"),
        use(
            RandomForestClassifier,
            n_estimators=Int(100, 500, initial=200),
        ).named("forest"),
    ).named("predictor")


def build_variants(feature_class=None):
    """Dériver des recettes indépendantes d'une base commune.

    feature_class peut désigner une classe métier, par exemple BMI. Elle est
    facultative pour pouvoir relire les variantes sans définir de composant.
    """
    base = preparation() >> models()

    compact = base.clone()
    compact.normalize.remove(RobustScaler, MaxAbsScaler).start("standard")
    compact.predictor.remove("forest")

    balanced = base.clone()
    balanced.add(optional(use(SMOTE)).named("imbalance"), before="predictor")
    balanced["forest"].configure(n_estimators=300)

    variants = {"base": base, "compact": compact, "balanced": balanced}

    if feature_class is not None:
        clinical = base.clone()
        clinical.add(use(feature_class).named("bmi"), before="cleaning")
        variants["clinical"] = clinical

    # Aucune de ces modifications n'altère base ou une autre variante.
    return variants


def compose_independent_pipelines():
    """Réutiliser une même déclaration dans plusieurs compositions."""
    shared_preparation = preparation()

    forest_pipeline = (
        shared_preparation
        >> use(RandomForestClassifier, n_estimators=200).named("predictor")
    )
    logistic_pipeline = (
        shared_preparation >> use(LogisticRegression).named("predictor")
    )

    forest_pipeline.normalize.start("standard")

    # La composition par >> copie les déclarations : shared_preparation et logistic_pipeline
    # gardent leur politique de démarrage. Aucun Step entraîné n'est partagé.
    return forest_pipeline, logistic_pipeline


# Utilisation :
# from iaml import IAML
# variants = build_variants(BMI)
# search = IAML(pipeline=variants["clinical"])
# search.fit(X, y)
