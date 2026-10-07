"""Naviguer parmi les variantes et configurer leurs paramètres.

find_all sélectionne une classe exacte ; un alias cible une seule variante.
Const verrouille une valeur, une valeur simple règle son départ optimisable."""

from iaml.flow import Const, Int, choice, use
from iaml.steps import (
    LogisticRegression,
    RandomForestClassifier,
    SimpleImputer,
    StandardScaler,
)


def build_pipeline():
    """Construire une recette dont les blocs et modèles sont nommés."""
    cleaning = use(SimpleImputer).named("cleaning")

    normalize = use(StandardScaler).named("normalize")

    predictor = choice(
        use(
            RandomForestClassifier,
            n_estimators=Int(100, 400, initial=200),
            max_depth=Int(3, 8, initial=5),
        ).named("forest_small"),
        use(
            RandomForestClassifier,
            n_estimators=Int(200, 600, initial=300),
            max_depth=Int(10, 30, initial=20),
        ).named("forest_deep"),
        use(LogisticRegression).named("logistic"),
    ).named("predictor")

    return cleaning >> normalize >> predictor


def inspect_pipeline(pipeline):
    """Récupérer une vue de l'arbre et la liste des alternatives du modèle."""
    models = [
        {"alias": option.alias, "component": option.component}
        for option in pipeline.predictor
    ]
    return {"tree": pipeline.describe(), "models": models}


def configure_pipeline(pipeline):
    """Éditer directement les objets attachés à la recette."""
    # find_all retourne toujours une sélection : zéro, une ou plusieurs briques.
    # La valeur simple change le départ sans remplacer les domaines :
    # forest_small garde [100, 400] ; forest_deep garde [200, 600].
    # Leurs domaines max_depth restent aussi distincts et inchangés.
    pipeline.predictor.find_all(RandomForestClassifier).configure(
        n_estimators=300,
    )

    # [] accepte uniquement un alias : cette sélection vise une brique précise.
    # Les alias étant uniques, l'accès est également possible depuis la racine.
    pipeline["forest_small"].configure(n_estimators=350)

    # Seul le départ de forest_small devient 350 ; son domaine reste [100, 400].
    return pipeline


def build_with_default_domain():
    """Changer le départ dès la construction, sans redéclarer les bornes."""
    # n_estimators reste optimisable dans son domaine par défaut.
    # max_depth reste fixé à 10 pendant toute la recherche.
    return use(
        RandomForestClassifier,
        n_estimators=300,
        max_depth=Const(10),
    ).named("forest_default_domain")


def build_parameter_variants():
    """Comparer un nouveau départ, une valeur fixe et un nouveau domaine."""
    base = use(
        RandomForestClassifier,
        n_estimators=Int(100, 500, initial=200),
    ).named("forest")

    new_start = base.clone()
    new_start.configure(n_estimators=300)
    # Domaine [100, 500] conservé ; départ 300, toujours optimisable.

    fixed = base.clone()
    fixed.configure(n_estimators=Const(300))
    # Valeur fixée à 300. Le domaine est conservé comme métadonnée inactive.

    resumed = fixed.clone()
    resumed.configure(n_estimators=350)
    # L'optimisation reprend dans [100, 500], avec un départ à 350.

    new_domain = base.clone()
    new_domain.configure(n_estimators=Int(200, 800, initial=400))
    # Nouveau domaine [200, 800] et nouveau départ 400.

    return {
        "base": base,
        "new_start": new_start,
        "fixed": fixed,
        "resumed": resumed,
        "new_domain": new_domain,
    }


# Utilisation :
# pipeline = configure_pipeline(build_pipeline())
# overview = inspect_pipeline(pipeline)
# print(overview["tree"])
# for model in overview["models"]:
#     print(model["alias"], model["component"])
#
# Contrats de cette API :
# - un alias explicite doit être unique dans toute la recette ;
# - [] sélectionne un alias ; find_all(Class) retourne une sélection iterable ;
# - Selection.configure(...) configure toutes les cibles et retourne la sélection ;
# - une valeur simple change le départ en conservant le domaine de chaque cible ;
# - Const(value) fixe uniquement le paramètre indiqué et conserve son domaine inactif ;
# - une valeur simple après Const réactive l'optimisation dans le domaine conservé ;
# - une valeur initiale hors domaine provoque une erreur, sans élargir les bornes ;
# - une constante est validée selon le composant, indépendamment des bornes de recherche ;
# - les paramètres sont validés sur toutes les cibles avant toute modification ;
# - un alias inconnu ou un paramètre inconnu provoque une erreur claire.
