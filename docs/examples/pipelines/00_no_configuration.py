"""Entraîner avec le preset AutoML par défaut, sans configuration.

Le preset contient ses stratégies principale et minimaliste. Aucun entraînement
n'est déclenché au chargement de ce module."""

from iaml import IAML


def train(X, y):
    """Recevoir les données de l'étude et retourner les candidats entraînés."""
    search = IAML()
    candidates = search.fit(X, y)
    return search, candidates
