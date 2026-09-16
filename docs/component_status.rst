Component availability
======================

``import iaml`` loads the components supported by the default search space.
``ActStandardScaler`` is registered under ``normalize``, and
``ActHistGradientBoostingRegressor`` under ``predictor``, ``tabular`` and
``regressor``. Their integration and real model fitting are covered by tests.
For the histogram regressor, the quantile search interval stays strictly inside
``(0, 1)``. As required by its estimator, ``loss='poisson'`` requires nonnegative
targets with a positive sum; an incompatible configuration raises an error.

The following modules are retained for explicit use or further development.
They are not re-exported by ``iaml`` or their component packages. Their steps
carry only the ``experimental`` tag, so importing one explicitly does not add
it to an automatic preprocessing, cleaning or prediction stage.

.. list-table:: Components excluded from automatic selection
   :header-rows: 1
   :widths: 28 20 52

   * - Component
     - Status
     - Reason
   * - ``ActCategoryStringToNumeric``
     - Incompatible prototype
     - Transforms the target rather than features, rebuilds its mapping on
       every call, and has no inverse transformation for predictions. Target
       encoding belongs with the predictor that requires it; ``ActCatBoost``
       already learns its own encoder and ``Predictor.predict`` reverses it.
   * - ``ActAalenAdditiveFitter``
     - Incomplete adapter
     - Requires ``lifelines``, which is absent from the standard dependencies.
       Its regularization parameter is not forwarded and predictions return
       a matrix of hazards instead of the risk scores expected by IAML.
       Existing stub-based tests do not validate a real lifelines fit.
   * - ``CumulativeDynamicAUCMetric``
     - Unvalidated metric
     - Passes survival probabilities where risk scores are required. The time
       grid and arithmetic averaging also need review. ``suitable`` returns
       ``False`` even after explicit import, preventing automatic selection.
       Its implementation is available for manual investigation only.
   * - ``ActDropBadQualityRows``
     - Experimental resampler
     - Missingness thresholds, the minimum sample policy and alignment of
       targets and groups need integration tests before automatic use.
       Explicit construction now initializes the standard step state.
   * - ``ActPolynomialFeatures``
     - Manual use
     - Polynomial expansion can create a very large feature matrix. Automatic
       exploration requires a bound on output size; the current implementation
       has none. Existing tests cover small, explicitly chosen inputs.
   * - ``ActCyclicalDateEncoding``
     - Manual use
     - Needs an explicit position while datetime columns are still present.
       The default pipeline runs date cleaning before feature preprocessing.
       Existing tests cover its direct transformations, not that integration.

Manual imports use the module path, for example::

   from iaml.actionables.features_preprocessing.act_polynomial_features import (
       ActPolynomialFeatures,
   )

An explicit import makes a component available to the caller; it does not
resolve the limitations listed above. ``ActDropBadQualityRows`` was formerly
re-exported despite missing standard step initialization; callers must now
import it from ``iaml.actionables.features_precleaning.act_drop_bad_quality_rows``.

Genetic search wrapper
----------------------

``WrapGeneticGridSearch`` uses ``Step.configuration``. It tracks outputs through
the current runner API and compares complete configurations when removing
duplicates. It no longer relies on ``learning_configuration`` or the legacy
``stacked_path`` history.

Predictors are not trained during candidate generation. To select subsequent
generations, pass an evaluator that returns a dictionary containing the
candidate's main metric. Use the original dataset, before pipeline transforms,
and the intended validation splitter::

   from iaml import Candidate, WrapGeneticGridSearch
   from iaml.actionables.predictors.regressor import ActDecisionTreeRegressor
   from iaml.metrics import R2ScoreMetric
   from iaml.splitters import kfold_splitter

   candidate = Candidate(dataset, metrics=[R2ScoreMetric()])

   def evaluate(candidate):
       return candidate.training_evaluate(dataset, splitter=kfold_splitter)

   search = WrapGeneticGridSearch(ActDecisionTreeRegressor(), evaluator=evaluate)
   search.configure({'nb_generations': 3, 'nb_estimators': 8})
   candidates = search.run(candidate)

Without an evaluator, several generations are supported only if the wrapped
step already supplies the main metric. A single generation can still produce
unevaluated candidates. The callback is a runtime dependency: when loading a
saved wrapper, supply it again with
``WrapGeneticGridSearch.from_pipeline(saved, evaluator=evaluate)``.
The normal IAML optimisation workflow can instead use ``GeneticOptimizer``,
which receives candidates scored by IAML.

Survival gradient boosting
--------------------------

``ActGradientBoostingSurvivalAnalysis`` has one implementation, in
``act_gradient_boosting_survival_analysis``, registered for both the normal
survival search and ``minimal_predictor`` selection. The former
``act_survival_xgboost`` module re-exports this same class for compatibility
with existing imports and pickled models; importing it does not register
another predictor. Both paths use scikit-survival's gradient boosting backend.
