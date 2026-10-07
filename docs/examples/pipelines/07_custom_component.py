"""Intégrer une brique métier avec le contrat Actionable/is_step.

La composition de recettes accepte les extensions IAML. La brique
BMI ci-dessous est placée explicitement dans la stratégie principale du preset."""

import pandas as pd

from iaml import Actionable
from iaml.decorators.is_step import is_step
from iaml.flow import PipelineSpec, use


@is_step("study_features")
class BodyMassIndex(Actionable):
    """Calculer le BMI à partir d'une taille en cm et d'un poids en kg."""

    name = "Body mass index"

    def __init__(self):
        self.optimizable = False

    def suitable(self, dataset):
        return "bmi" not in dataset.X and all(
            column in dataset.X
            and pd.api.types.is_numeric_dtype(dataset.X[column])
            for column in ("height_cm", "weight_kg")
        )

    def fit(self, dataset):
        return self

    def transform(self, X):
        result = X.copy()
        height_m = result["height_cm"].where(result["height_cm"] > 0) / 100
        weight_kg = result["weight_kg"].where(result["weight_kg"] > 0)
        result["bmi"] = weight_kg / height_m**2
        return result


def build_pipeline():
    pipeline = PipelineSpec.default()
    pipeline.add(use(BodyMassIndex).named("bmi"), before="cleaning")
    return pipeline


# Le tag métier évite que le preset ajoute déjà cette brique automatiquement.
# Avec le tag "features_precleaning", un simple import la rendrait au contraire
# disponible dans la famille automatique correspondante, sans placement manuel.
