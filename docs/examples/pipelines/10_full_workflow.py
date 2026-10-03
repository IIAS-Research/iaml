"""Configurer le pipeline, les métriques, le descriptif et les explications.

Les collections analytiques calculent les méthodes applicables sans multiplier
les candidats. pos_label, k et nsamples restent fixes. Aucun calcul statistique
ni entraînement n'est déclenché au chargement de ce module."""

from iaml import (
    AccuracyMetric,
    ConfusionMatrixPlot,
    CorrelationWithTargetStatistic,
    F1ScoreMetric,
    IAML,
    RecallMetric,
    TopKValueCountsStatistic,
)
from iaml.explainers import KernelSHAP
from iaml.flow import PipelineSpec, explanations, metrics, statistics, use
from iaml.steps import MaxAbsScaler, UnitNormScaler


def build_search():
    """Adapter une étude sans reconstruire son pipeline AutoML par défaut."""
    search = IAML(
        pipeline=PipelineSpec.default(),
        main_metric=F1ScoreMetric(pos_label=1),
        metrics=metrics().remove(AccuracyMetric),
        statistics=statistics().remove(CorrelationWithTargetStatistic),
        explanations=explanations(
            use(ConfusionMatrixPlot).named("confusion"),
            use(KernelSHAP, nsamples=100).named("shap"),
        ),
    )

    # Même navigation et mêmes opérations sur les objets attachés à la recette.
    search.pipeline.normalize.remove(UnitNormScaler, MaxAbsScaler)
    search.metrics.find_all(RecallMetric).configure(pos_label=1)
    search.statistics.find_all(TopKValueCountsStatistic).configure(k=5)

    # Le nom reste local à cette collection ; aucun IAML["alias"] n'est ajouté.
    search.explanations["shap"].configure(nsamples=100)

    # La métrique principale reste présente avec sa configuration. Une exclusion
    # qui la retire est une erreur avant fit ; elle n'est pas réintroduite
    # implicitement.
    return search


def run_study(X_train, y_train, X_test, y_test):
    """Analyser une classification binaire 0/1 avec un test tenu hors recherche.

    X_train/X_test sont des DataFrames et y_train/y_test des Series, avec 1 comme
    événement étudié. Les objets retournés peuvent être affichés ou sauvegardés
    par l'appelant ; les explications sont calculées seulement sur les cinq lignes
    explicitement demandées.
    """
    search = build_search()

    # Forme explicite de la méthode descriptive : décrire les données fournies
    # avant tout modèle, sans faire de ces calculs une étape d'entraînement.
    training_statistics = search.get_descriptive_statistics(X_train, y_train)

    model = search.fit(X_train, y_train)[0]

    # Sans argument, décrire le dernier dataset utilisé pour la recherche,
    # potentiellement échantillonné. Il peut différer du tableau ci-dessus.
    search_statistics = search.get_descriptive_statistics()

    # Les scores internes classent les candidats ; les scores de test sont une
    # nouvelle évaluation et ne remplacent pas les résultats de validation.
    validation_scores = model.describe_metrics()
    test_scores = model.evaluate(X_test, y_test)

    # Appel collectif : exécuter les méthodes de la collection
    # applicables au modèle et aux données, puis retourner leurs résultats par
    # alias. La matrice de confusion utilise y ; SHAP utilise le modèle et X.
    # nsamples=100 règle l'effort SHAP par prédiction, pas le nombre de patients.
    explanation_results = model.explain(X_test.iloc[:5], y_test.iloc[:5])

    # Les méthodes spécialisées actuelles restent disponibles avec leur contrat,
    # notamment explain_feature_importance(X, nsamples=...) -> Explanation.
    # Ce nouvel appel collectif ne modifie pas silencieusement leur retour.
    return {
        "search": search,
        "model": model,
        "training_statistics": training_statistics,
        "search_statistics": search_statistics,
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "explanations": explanation_results,
    }
