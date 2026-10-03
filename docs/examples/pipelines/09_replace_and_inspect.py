"""Remplacer un bloc, décrire sa recette et reconstruire son code.

Les rapports describe et diff sont structurés et affichables. Aucun
entraînement n'est lancé avant l'appel explicite de train_and_inspect."""

from iaml import IAML
from iaml.flow import Const, PipelineSpec, normalizers
from iaml.steps import MaxAbsScaler, UnitNormScaler, RandomForestClassifier


def build_variant():
    """Remplacer un bloc avec la même grammaire que sa construction."""
    base = PipelineSpec.default()
    pipeline = base.clone()
    old_normalize = pipeline.normalize

    new_normalize = old_normalize.replace(
        normalizers().remove(UnitNormScaler, MaxAbsScaler)
    )
    # replace conserve la position et l'alias "normalize" du bloc ciblé.
    # Le bloc de remplacement est copié puis attaché à cette recette ; la
    # méthode retourne ce nouveau bloc attaché, également pipeline.normalize.
    # old_normalize devient détaché : ses futures éditions n'affectent plus
    # pipeline. Une ancienne référence ne vise pas magiquement le nouveau bloc.
    # L'alias de la racine du remplacement doit être absent ou "normalize" ;
    # un alias différent est refusé. Les alias descendants restent uniques.

    pipeline.predictor.find_all(RandomForestClassifier).configure(
        n_estimators=300,
        max_depth=Const(10),
    )

    return base, pipeline, new_normalize


def inspect_recipe(pipeline, base):
    """Inspecter le plan déclaré avant le lancement d'une recherche."""
    return {
        "description": pipeline.describe(),
        "changes": pipeline.diff(base),
        "code": pipeline.to_code(),
    }


def train_and_inspect(X, y):
    """Comparer l'intention déclarée au rapport du lancement effectif."""
    base, pipeline, _ = build_variant()
    recipe_overview = inspect_recipe(pipeline, base)
    search = IAML(pipeline=pipeline, max_duration=60)
    result = search.fit(X, y)
    run_overview = search.describe()
    return search, result, recipe_overview, run_overview


# Contrats d'inspection :
# - PipelineSpec.describe() décrit l'arbre déclaré, les alternatives autorisées,
#   celles présentes au démarrage, les exclusions/ajouts et les paramètres :
#   domaine, valeur initiale ou mode fixe et domaine conservé mais inactif ;
# - une famille issue du registre est indiquée comme non résolue avant fit ;
#   describe ne promet pas de déterminer statiquement toutes les compatibilités,
#   car suitable() peut dépendre des transformations précédentes ;
# - IAML.describe(), après fit, décrit la recette effectivement compilée et les
#   familles résolues pour ce lancement, avec les décisions d'applicabilité
#   observées, leurs raisons disponibles et les candidats réellement évalués ;
# - diff(base) décrit les différences déclaratives de cette recette par rapport
#   à base : blocs, alternatives, politiques de démarrage et paramètres ;
# - to_code() produit le code de reconstruction de la recette déclarative,
#   avec les alias, domaines, constantes et politiques explicites. Pour une
#   brique métier, il référence sa classe ; il ne copie pas son implémentation ;
# - le rapport du lancement conserve la version et le catalogue résolu/figé.
#   Le code d'une recette contenant une famille ouverte exprime cette famille,
#   pas la promesse d'un catalogue identique dans un autre environnement ;
# - cette reproductibilité déclarative ne garantit pas des scores identiques
#   sans conserver aussi données, versions et état aléatoire de l'entraînement.
