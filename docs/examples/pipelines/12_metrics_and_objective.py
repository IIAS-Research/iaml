"""Distinguer les variantes d'une métrique et sélectionner l'objectif par alias.

Les paramètres de calcul, notamment pos_label, restent fixes. Les variantes
possèdent des clés distinctes ; aucun calcul n'est lancé à l'import."""

from iaml import AccuracyMetric, IAML, RecallMetric
from iaml.flow import metrics, use


def build_metrics():
    """Mesurer séparément le rappel de l'événement et celui de l'autre classe."""
    return metrics(
        use(RecallMetric, pos_label=1).named("event_recall"),
        use(RecallMetric, pos_label=0).named("other_recall"),
        use(AccuracyMetric).named("accuracy"),
    )


def build_search():
    """Conserver le pipeline AutoML par défaut et choisir un objectif nommé."""
    search = IAML(
        metrics=build_metrics(),
        main_metric="event_recall",
    )

    # [] cible un alias et préserve la configuration de l'autre variante.
    # Ici, 1 désigne explicitement l'événement étudié dans une cible binaire 0/1.
    search.metrics["event_recall"].configure(pos_label=1)

    # find_all retourne les deux variantes, également consultables sans boucle
    # de configuration qui leur imposerait accidentellement le même pos_label.
    recall_variants = search.metrics.find_all(RecallMetric)
    return search, recall_variants


def without_secondary_recall():
    """Construire une variante indépendante en supprimant une mesure secondaire."""
    original = build_metrics()
    reduced = original.clone()
    reduced.remove("other_recall")
    return original, reduced


def run_study(X_train, y_train, X_test, y_test):
    """Comparer validation interne et test sur une classification binaire 0/1.

    X_train/X_test sont des DataFrames et y_train/y_test des Series alignées ;
    1 représente l'événement et 0 l'autre classe. Les données de test ne sont
    pas utilisées pour sélectionner les candidats ni régler leurs paramètres.
    """
    search, _ = build_search()
    model = search.fit(X_train, y_train)[0]

    # Les scores de validation classent les candidats selon event_recall.
    # L'évaluation de test calcule les mêmes définitions de métriques et conserve
    # leurs alias ; elle ne remplace pas les scores de validation stockés.
    validation_scores = dict(model.computed_metrics)
    test_scores = model.evaluate(X_test, y_test)

    return {
        "search": search,
        "model": model,
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "event_recall_on_test": test_scores["event_recall"],
        "other_recall_on_test": test_scores["other_recall"],
    }


# Contrats des métriques et de leur compilation :
# - main_metric conserve son contrat historique d'instance Metric et accepte
#   aussi l'alias d'une brique présente dans la collection déclarative metrics ;
# - le compilateur résout cet alias vers la définition exacte de la métrique,
#   avec ses paramètres, puis transmet cette définition à Candidate.metrics ;
# - la compilation porte la clé/alias jusqu'à computed_metrics, evaluate
#   et describe_metrics, sans changer compute, needed_prediction, suitable
#   ni greater_is_better. Les variantes aux clés indistinguables sont refusées
#   avant le calcul et ne s'écrasent jamais silencieusement ;
# - event_recall et other_recall restent deux entrées distinctes dans les scores,
#   même s'ils reposent sur la même classe ; les alias sont uniques dans cette
#   collection. [] utilise ces alias et find_all(RecallMetric) retrouve les deux ;
# - la métrique principale, ses paramètres et le sens de classement sont figés
#   pour toute la recherche. Une modification de la recette s'applique à un
#   nouveau lancement, pas au classement des candidats déjà en cours ;
# - un objectif inconnu, retiré de la collection avant fit ou inapplicable au
#   problème provoque une erreur de validation ; la façade ne le réintroduit
#   pas implicitement. Retirer other_recall laisse event_recall disponible ;
# - aucun domaine Int/Float ni Const n'est nécessaire pour pos_label : il s'agit
#   d'un paramètre de calcul fixe, distinct des paramètres AutoML du pipeline.
