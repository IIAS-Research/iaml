"""Calculer un descriptif configurable avant toute recherche de modèle.

Les statistiques lisent les données explicitement fournies sans les transformer
ni remplacer le dernier dataset de recherche mémorisé par IAML."""

from iaml import (
    IAML,
    MissingRateStatistic,
    SummaryTableStatistic,
    TopKValueCountsStatistic,
)
from iaml.flow import statistics, use


def build_search():
    """Choisir les analyses utiles, sans modifier le pipeline par défaut."""
    return IAML(
        statistics=statistics(
            SummaryTableStatistic,
            MissingRateStatistic,
            use(TopKValueCountsStatistic, k=5).named("categories"),
        ),
    )


def describe_dataset(X, y):
    """Décrire les données fournies sans appeler fit."""
    search = build_search()
    search.statistics["categories"].configure(k=10)
    # k est un paramètre de calcul fixe, pas un hyperparamètre à optimiser.
    table = search.get_descriptive_statistics(X, y)
    return search, table


def replace_description(X, y):
    """Remplacer tout le bloc avec la syntaxe utilisée pour le construire."""
    search = build_search()
    search.statistics.replace(
        statistics(SummaryTableStatistic, MissingRateStatistic),
    )
    return search.get_descriptive_statistics(X, y)


# Contrat descriptif :
# - statistics(...) contient toutes les méthodes à calculer, pas des alternatives ;
# - les méthodes sont filtrées selon leur applicabilité aux données et à la cible ;
# - X et y désignent explicitement le dataset décrit, indépendamment d'un fit ;
# - cet appel ne remplace pas le dataset mémorisé par un précédent fit ;
# - sans arguments, get_descriptive_statistics() décrit le dernier dataset brut
#   de recherche, éventuellement échantillonné, comme dans l'IAML actuel ;
# - aucun descriptif d'un état intermédiaire de preprocessing n'est promis ;
# - fournir y avec X : toutes les statistiques actuelles
#   ne supportent pas l'absence de cible ;
# - les caches descriptifs doivent tenir compte des données et de la configuration.
