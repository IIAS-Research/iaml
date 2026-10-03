.. _pipelines-guide:

==========================
Build and adapt a pipeline
==========================

IAML pipelines are editable recipes: combine reusable components, choose the
alternatives the search may explore, and configure their parameters. You can
start from the AutoML preset or describe a pipeline from scratch. The same
operations work when you revisit an existing recipe.

For metrics, descriptive statistics and explanations, see :doc:`study`.
The complete signatures and validation rules are in :doc:`reference`.

Start from the preset or your own recipe
========================================

The :doc:`integrated workflow <../quick_start>` trains without a recipe.
To adjust its preset, obtain an editable copy:

.. code-block:: python

   from iaml import IAML
   from iaml.flow import PipelineSpec
   from iaml.steps import MaxAbsScaler, UnitNormScaler

   pipeline = PipelineSpec.default()
   pipeline.main.normalize.remove(UnitNormScaler, MaxAbsScaler)
   search = IAML(pipeline=pipeline)

``IAML().pipeline`` and ``PipelineSpec.default()`` expose the same preset, with
two visible branches. ``pipeline.main`` contains the main strategy;
``pipeline.minimal`` contains the minimal strategy, whose models are available
through ``pipeline.minimal.minimal_predictor``. Remove the latter strategy with
``pipeline.remove("minimal")``. Editing a group in one branch leaves the other
branch independent. Unique aliases also allow direct access to groups such as
``pipeline.normalize`` and ``pipeline.predictor``.

The main strategy starts without feature selection. Its ``features_selection``
choice can use one selector at a time, including ``SelectPercentile``. Custom
recipes can still declare successive selections explicitly.

An explicitly supplied pipeline defines the requested search scope: IAML
generates only its declared branches. No additional minimal candidates are
added to a custom recipe. Omitting ``pipeline`` selects the complete preset,
even when you configure only the analytical collections.

``IAML(pipeline=pipeline)`` copies the recipe. To edit the study after
construction, navigate from ``search.pipeline`` rather than the original
``pipeline``. Each ``fit`` creates independent execution instances.

.. _pipeline-guide-composition:

Compose components with ``>>``
==============================

``use(Component, **params)`` describes a component. Import the IAML adapters
from ``iaml.steps`` using names such as ``StandardScaler`` and
``RandomForestClassifier``. Historical imports with the ``Act`` prefix remain
available.

.. code-block:: python

   from iaml.flow import Int, choice, optional, use
   from iaml.steps import (
       LogisticRegression,
       RandomForestClassifier,
       RobustScaler,
       SimpleImputer,
       SMOTE,
       StandardScaler,
   )

   normalization = choice(StandardScaler, RobustScaler).named("normalize")
   models = choice(
       use(LogisticRegression).named("logistic"),
       use(
           RandomForestClassifier,
           n_estimators=Int(100, 500, initial=200),
       ).named("forest"),
   ).named("predictor")

   pipeline = (
       use(SimpleImputer).named("cleaning")
       >> normalization
       >> optional(use(SMOTE)).named("imbalance")
       >> models
   )

Read ``a >> b`` as "a, then b": candidates produced by ``a`` continue through
``b`` in that order. The operator composes recipes, so write
``use(A) >> use(B)``. A class supplied directly to ``choice`` is equivalent to
``use(Class)``.

``choice`` declares allowed alternatives. By default, each participates in
initial candidate generation. The example allows two normalizations, with or
without SMOTE, followed by two models: eight possible structures before
applicability checks. The budget and compatibility with the data determine
which are generated and evaluated. This structural scope does not guarantee
exhaustive hyperparameter search.

``optional`` also permits the absence of its transformation or resampler.
Every modeling path must end in exactly one predictor; a predictor cannot be
made optional.

Some preset groups use adaptive ordering: their components run according to
their priority. Editing such a group with ``add`` or ``remove`` preserves this
mode. An adaptive group executes its applicable components rather than
selecting one model alternative.

Start with a family, then subtract
==================================

Families let you use registered components without enumerating them. Remove
the components outside your intended search scope:

.. code-block:: python

   from iaml.flow import normalizers, predictors
   from iaml.steps import KNeighborsClassifier, MaxAbsScaler, UnitNormScaler

   normalization = (
       normalizers()
       .remove(UnitNormScaler, MaxAbsScaler)
       .named("normalize")
   )
   models = predictors().remove(KNeighborsClassifier).named("predictor")

   pipeline = use(SimpleImputer).named("cleaning") >> normalization >> models

``normalizers`` and ``predictors`` discover the corresponding component
families. ``choice(tag="normalize")`` provides the generic form using a registry
tag. ``remove(Class)`` removes that exact class's variants from the targeted
group; ``remove("alias")`` removes the named variant. An explicit ``add`` can
introduce a configured variant of a previously excluded component, without
restoring its other registry variants.

Families resolve when a search starts. The resulting catalogue, including
exclusions, additions and variant configurations, is frozen for that run.
Component replacements by the optimizer respect this scope. Each component's
applicability is still checked against the data it receives.

.. _pipeline-guide-selection:

Target one variant or several
=============================

A component class can occur several times with different configurations.
Aliases distinguish its variants:

.. code-block:: python

   models = choice(
       use(
           RandomForestClassifier,
           n_estimators=Int(100, 400, initial=200),
       ).named("forest_small"),
       use(
           RandomForestClassifier,
           n_estimators=Int(200, 600, initial=300),
       ).named("forest_deep"),
   ).named("predictor")

   models.find_all(RandomForestClassifier).configure(n_estimators=300)
   # Each variant retains its own domain.

   models["forest_small"].configure(n_estimators=350)
   # Only this variant's initial value changes.

``find_all`` matches the exact class and returns an iterable ``Selection``.
Its ``configure`` method validates all targets before modifying them; each
variant retains its own unspecified settings and domain. The selection is a
snapshot: query again after removing or replacing a target. See
:ref:`the editing contracts <pipelines-ref-editing>` for stale selections,
empty groups and strict removal behavior.

Named access returns objects attached to the recipe. Thus,
``models["forest_small"].configure(...)`` edits that variant inside ``models``.
``[]`` accepts an alias; class selection uses ``find_all``.

An explicit alias is unique throughout a pipeline tree, so it can be used
from the root without a qualified path. Name variants individually with
``.named(...)``.

.. _pipeline-guide-editing:

Add, remove and replace
=======================

Editing uses the same components as construction:

.. code-block:: python

   from iaml.flow import PipelineSpec, normalizers
   from iaml.steps import KNeighborsClassifier, MaxAbsScaler, UnitNormScaler

   pipeline = PipelineSpec.default()

   pipeline.predictor.remove(KNeighborsClassifier).add(
       use(
           RandomForestClassifier,
           n_estimators=Int(50, 150, initial=100),
       ).named("forest_small"),
   )

   pipeline.remove("imbalance")

   previous = pipeline.normalize
   replacement = previous.replace(
       normalizers().remove(UnitNormScaler, MaxAbsScaler),
   )

``add`` and ``remove`` edit the targeted group: alternatives in a choice,
steps in a sequence, or methods in an analytical collection. In a sequence,
``before="alias"`` specifies where to insert a component. The alias can belong
to a nested sequence: ``pipeline.add(use(BMI).named("bmi"), before="cleaning")``
inserts the component into the preset's ``main`` strategy, just before cleaning.

Edits are atomic. A group may be empty while you edit it, but the completed
recipe must have one terminal predictor per path before training.

``replace`` returns a new attached block, preserving the old block's position
and alias. In the example, ``replacement`` is now ``pipeline.normalize``;
``previous`` is detached and no longer edits the pipeline. Continue from the
returned replacement. See :ref:`pipelines-ref-editing` for replacement
validation and container requirements.

.. _pipeline-guide-parameters:

Configure parameters
====================

The same notation works in ``use`` and ``configure``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Value
     - Effect on a training component
   * - ``300``
     - Change the initial value, retaining the current or default domain.
   * - ``Const(300)``
     - Keep the value fixed throughout the search.
   * - ``Int(100, 500, initial=300)``
     - Set an integer domain and its initial value.
   * - ``Float(0.01, 10, initial=1)``
     - Set a real-valued domain and its initial value.

.. code-block:: python

   from iaml.flow import Const, Int

   forest = use(
       RandomForestClassifier,
       n_estimators=Int(100, 500, initial=200),
       max_depth=Const(10),
   )

   forest.configure(n_estimators=300)
   # Start at 300 within [100, 500]; depth stays fixed at 10.

   forest.configure(n_estimators=Const(300))
   # Fix the value at 300; retain [100, 500] as an inactive domain.

   forest.configure(n_estimators=350)
   # Resume optimization within [100, 500], starting at 350.

   forest.configure(n_estimators=Int(200, 800, initial=400))
   # Replace the domain and its initial value.

Omitted parameters retain their configuration. An initial value outside its
active domain is rejected without widening the bounds. A constant must satisfy
the component's own constraints, but may lie outside a retained, inactive
search domain. If a parameter has no domain, supplying a plain value does not
invent one; the component's parameter policy remains in effect.

This convention also applies to resamplers. Analytical parameters configure
fixed calculations instead; see :ref:`pipelines-study-collections`.

.. _pipeline-guide-start:

Choose the starting alternatives
================================

``start`` selects the alternatives initially generated by a choice, while
keeping the other alternatives available. Give individual alternatives aliases
with ``named``:

.. code-block:: python

   models = choice(
       use(LogisticRegression).named("logistic"),
       use(RandomForestClassifier).named("forest"),
   ).named("predictor")

   complete = use(SimpleImputer) >> models

   partial = complete.clone()
   partial.predictor.start("logistic")

   restricted = complete.clone()
   restricted.predictor.remove("forest")

``complete`` allows both models and starts with both. ``partial`` still allows
both, but starts with logistic regression; an optimizer that can replace
components can subsequently reach the forest. ``restricted`` excludes the
forest throughout the run.

``partial.predictor.start()`` restores a full start. Before removing an
alternative referenced by ``start``, reset the starting alternatives with
``start()`` or select a start that no longer references it. Otherwise removal
fails without changing the group.

A partial start requires an optimizer supporting component replacement,
such as ``GeneticOptimizer``. Choose the optimizer and validation settings in
:ref:`pipelines-study-search`. Choices between complete sub-pipelines use
full initial generation.

Keep preparation and models together
====================================

Independent preparation and model choices allow all their combinations. To
keep a preparation strategy associated with its model, choose between complete
sub-pipelines instead:

.. literalinclude:: ../examples/pipelines/08_coherent_strategies.py
   :language: python
   :pyobject: build_pipeline

The ``linear`` branch normalizes the data before logistic regression. The
``forest`` branch uses its own random forest configuration and search domains.
Both share the preceding cleaning step. Nested composition expresses this
structural relationship directly.

Their parameters are optimized within their respective domains, while the
optimizer preserves each strategy's component scope. These complete branches
use full initial generation; whole-strategy replacement during partial search
is unsupported.

Reuse recipes independently
===========================

Composition with ``>>`` and ``clone`` produce independent recipes. Group
constructors, ``IAML(pipeline=recipe)``, ``add(recipe)`` and ``replace(recipe)``
also copy their inputs. After construction, edit the attached objects reached
through the new container:

.. code-block:: python

   preparation = (
       use(SimpleImputer).named("cleaning")
       >> normalizers().named("normalize")
   )

   forest_pipeline = preparation >> use(RandomForestClassifier).named("predictor")
   logistic_pipeline = preparation >> use(LogisticRegression).named("predictor")

   forest_pipeline.normalize.remove(UnitNormScaler)
   # preparation and logistic_pipeline remain independent.

   fixed_depth = forest_pipeline.clone()
   fixed_depth.predictor.configure(max_depth=Const(10))

A reference retained before composition points to its original fragment.
Navigate from the final pipeline, or use its aliases, to edit the composed
copy. Recipe edits after training prepare a later run and do not change
existing fitted candidates.

Integrate a custom component
============================

Extensions use IAML's existing component contract: registration with
``is_step``, applicability checks, and transformation or prediction methods.
Pass the registered class to ``use``:

.. code-block:: python

   def with_business_feature(feature_class):
       pipeline = PipelineSpec.default()
       pipeline.add(use(feature_class).named("bmi"), before="cleaning")
       return pipeline

The :download:`custom component example
<../examples/pipelines/07_custom_component.py>` contains a complete body mass
index component and its insertion. Its registry tag also determines which
automatic families discover it. See :doc:`../adaptability` for component
contracts and other extension points.

Inspect a recipe and its search
===============================

.. code-block:: python

   base = PipelineSpec.default()
   pipeline = base.clone()
   pipeline["normalize"].remove(UnitNormScaler)

   print(pipeline.describe())
   print(pipeline.diff(base))
   code = pipeline.to_code()

``describe`` shows the tree, allowed and initial alternatives, exclusions, and
parameters with their domains or fixed-value mode. ``diff`` reports changes
relative to another recipe. Both return structured, readable ``RecipeReport``
objects: print the report or obtain a copy of its data with ``to_dict()``.
``to_code`` generates reconstruction code, preserving aliases, constants and
both active and inactive domains.

Before ``fit``, open families are unresolved. Afterward, ``search.describe()``
reports the frozen catalogue, observed applicability and evaluated candidates.
Save this report alongside reconstruction code: an open family can resolve
to a different catalogue in another environment. See
:ref:`pipelines-ref-inspection` for report fields and reconstruction limits.

See :doc:`reference` for signatures and diagnostics.
