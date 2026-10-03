:orphan:

.. _pipelines-ref:

Pipeline and study API reference
================================

Contracts for the recipe constructors, editing methods and study interfaces.
For a hands-on introduction, read :doc:`guide` and study; for individual
Python classes, see :doc:`../autoapi/index`.

Recipes declare configuration and never hold fitted estimators. Training
returns ``Candidate`` objects containing fitted ``IAMLPipeline`` instances.

.. _pipelines-ref-functions:

Construction functions
----------------------

Composition functions and parameter domains are available from ``iaml.flow``.
The names in ``iaml.steps``, such as ``StandardScaler`` and
``RandomForestClassifier``, refer to IAML component adapters, rather than raw
scikit-learn estimators. The historical imports remain supported.

``UnitNormScaler`` scales each nonzero row so its L1 or L2 norm is 1.
It uses the L2 norm by default and leaves zero rows unchanged.
``Normalizer`` remains an alias for the same component for compatibility.
The ``normalizers()`` function refers to the full family of scaling methods.

.. list-table:: Public constructors
   :header-rows: 1
   :widths: 32 68

   * - Signature
     - Result and purpose
   * - ``use(Component, **params)``
     - Recipe for one component, configured with its declared parameters.
       Does not run a computation.
   * - ``Int(low, high, initial=...)``
     - Bounded integer search domain and initial value for a training parameter.
   * - ``Float(low, high, initial=...)``
     - Bounded real search domain and initial value for a training parameter.
   * - ``Const(value)``
     - Value held fixed throughout the search.
   * - ``choice(*alternatives, tag=None)``
     - Explicit alternatives or a family from the component registry.
       These construction modes are mutually exclusive.
   * - ``normalizers()``
     - Registered normalization family, editable as a choice.
   * - ``predictors()``
     - Registered predictor family, filtered at execution according to
       applicability to the task and data.
   * - ``optional(recipe)``
     - Recipe allowing either the supplied transformation or its absence.
   * - ``PipelineSpec.default()``
     - Independent recipe for the built-in preset, with ``main`` and ``minimal``.
   * - ``metrics(*components)``
     - Metric collection; with no arguments, the default collection.
   * - ``statistics(*components)``
     - Descriptive collection; with no arguments, the default collection.
   * - ``explanations(*components)``
     - Explanation and performance-plot collection; with no arguments, the
       default collection.

``use`` accepts training and analytical components. It preserves each
component's role: statistics describe data, metrics measure predictions and
explanations operate on fitted models. Analytical components do not become
preprocessing steps.

Alternatives and collections also accept a component class directly,
equivalent to ``use(Class)``. Use ``use`` when a component needs parameters or
an alias. Raw classes cannot be composed with each other using ``>>``.

.. _pipelines-ref-composition:

Composition and search scope
----------------------------

``left >> right`` creates a sequence in the written order. Each candidate
produced on the left continues through the recipe on the right. The operator
copies its recipe fragments: it leaves its operands unchanged and shares no
mutable configuration with the resulting sequence.

``choice(A, B)`` allows alternatives ``A`` and ``B``, which can be individual
components or complete subpipelines. It explores alternative structures; it
does not execute parallel transformations and merge their outputs.

By default, all declared alternatives participate in initial generation,
subject to applicability. This guarantees neither exhaustive hyperparameter
search nor completed evaluation of every branch within the time budget.
Successive choices combine their structural possibilities.

``choice(tag="...")`` declares a registered family. The family is resolved at
launch, accounting for exclusions and explicit additions. Its resolved scope
remains fixed for that launch, including component replacements performed by
the optimizer. An excluded class cannot return implicitly from the global
registry. Runtime ``suitable()`` checks still apply to the data encountered
at each step.

``optional`` applies to a transformation, including resampling such as SMOTE.
It allows a branch without that transformation. It cannot remove the predictor
required to make a complete candidate.

``PipelineSpec.default()`` exposes the preset's named blocks, including
``cleaning``, ``normalize``, ``imbalance`` and ``predictor``. Groups retain their
mode when edited: an ordered sequence, a choice of alternatives or an adaptive
group whose components execute according to priority. Editing an adaptive
group does not implicitly turn it into a choice.

.. _pipelines-ref-aliases:

Naming and navigation
---------------------

``recipe.named(alias)`` assigns an alias and returns the named object for
chaining. Names are not keyword arguments to ``choice`` or ``add``. Aliases
are optional until you need to address an individual component or distinguish
analytical outputs.

Explicit aliases are unique throughout one pipeline recipe tree. Each
analytical collection has its own namespace: a metric alias may also appear
in the descriptive collection. There is no global ``IAML["alias"]`` lookup.

``pipeline["alias"]`` searches the pipeline tree. ``group["alias"]`` searches
the targeted group's subtree. Unknown names raise an error. Unique aliases
make hierarchical paths unnecessary. Classes are not accepted as ``[]``
selectors.

Attribute access to preset blocks, such as ``pipeline.normalize`` and
``pipeline.predictor``, returns their attached objects. ``[]`` provides an
explicit lookup for user aliases, including names that are not valid Python
attributes or collide with a recipe method.

Groups are iterable. Individual component recipes expose ``alias`` and
``component``. ``find_all(Component)`` returns an iterable ``Selection`` with
zero, one or more matching recipes in the targeted scope. A class selects
all its exact-class variants, rather than implicitly selecting one variant.

.. _pipelines-ref-editing:

Editing methods
---------------

Editing methods mutate the targeted recipe; ``clone`` and inspection methods
return independent objects or reports. Objects obtained by navigation are
attached to their containing recipe, so editing them updates that recipe.
These edits do not affect an already compiled or trained candidate.

Edits are atomic: validation precedes mutation, and a rejected operation
leaves the recipe unchanged. Constructors, composition, ``add``, ``replace``
and ``clone`` copy the supplied recipe fragments.

.. list-table:: Methods and their scope
   :header-rows: 1
   :widths: 30 50 20

   * - Signature
     - Effect
     - Return value
   * - ``named(alias)``
     - Assigns an alias to the targeted object, checking uniqueness.
     - Targeted object.
   * - ``configure(**params)``
     - Configures the component; unspecified parameters retain their values.
     - Targeted component.
   * - ``find_all(Component)``
     - Selects exact-class variants in the targeted scope.
     - ``Selection``.
   * - ``Selection.configure(**params)``
     - Validates all targets, then configures each independently.
     - Same selection.
   * - ``add(recipe, before=None)``
     - Adds a step to a sequence or an alternative to a choice. ``before``
       identifies a named anchor in a sequence, including a nested preset
       sequence.
     - Targeted group.
   * - ``remove(*selectors)``
     - Removes local occurrences of a class or the named block within the
       targeted group.
     - Targeted group.
   * - ``replace(recipe)``
     - Replaces the attached block at its position with a copy of the supplied
       recipe.
     - New attached block.
   * - ``clone()``
     - Copies content, aliases and configuration independently.
     - New recipe.
   * - ``start(*aliases)``
     - Selects a choice's initial alternatives without narrowing its allowed
       alternatives.
     - Targeted choice.

``add`` and ``remove`` preserve the group's mode. Adding a variant of an
excluded class authorizes that explicit variant without clearing the registry
exclusion. Its alias comes from ``.named(...)``. With ``before``, insertion
occurs in the sequence containing the anchor, even when the method is called
from the preset root.

``find_all(Class)`` and ``remove(Class)`` match the exact class, without
including subclasses. Removing an absent class raises ``ValueError``;
removing an unknown alias raises ``KeyError``. Removing multiple selectors
is atomic.

A saved selection is a snapshot and does not include later additions. If a
target is removed or replaced, the selection becomes stale and fails before
any collective modification; call ``find_all`` again. A group may remain
temporarily empty while you edit it, but an empty choice is rejected before
training. You can attach an incomplete recipe with ``IAML(pipeline=...)``
and finish it through ``search.pipeline`` before ``fit``.

``replace`` preserves the replaced block's position and alias. The supplied
recipe's root must be unnamed or carry that same alias; a different root
alias is rejected. Descendant aliases are checked in the resulting recipe.
The supplied content is copied, and its mode becomes the attached block's
mode. Analytical replacements must belong to the same family.

``replace`` requires a container, including when called on an old, detached
reference. A root attached to ``IAML`` can be replaced.

The old block becomes detached: editing it no longer changes the containing
recipe. A reference saved before composition similarly addresses its original
fragment. Navigate from the composed recipe to edit the copy. See
:ref:`pipeline-guide-editing` for a replacement example.

``start`` belongs to choices and accepts alternative aliases, rather than
parameter values. A partial start requires an optimizer capable of replacing
components to reach alternatives omitted from initial generation. An optimizer
that only searches parameter values rejects a partial start.

Partial exploration supports choices of components at the same location.
Atomic replacement of whole subpipelines is not supported. Call ``start()``
without arguments to restore full initial generation. Removing an alternative
referenced by ``start`` is rejected: reset the start with ``start()`` before
removing it, or explicitly select a different set of initial alternatives.

.. _pipelines-ref-parameters:

Parameters and domains
----------------------

Training components support three parameter forms in both ``use`` and
``configure``:

.. list-table:: Training-parameter configuration
   :header-rows: 1

   * - Form
     - Value and domain
     - Optimization
   * - ``n_estimators=300``
     - Initial value 300; existing or default domain retained.
     - Component's AutoML behavior, reactivated if the parameter was fixed.
   * - ``n_estimators=Const(300)``
     - Value 300; domain retained as inactive metadata.
     - Fixed parameter.
   * - ``n_estimators=Int(100, 500, initial=300)``
     - Integer domain [100, 500] and initial value 300.
     - Optimizable within this domain.

``Float`` follows the same rules for real values. Bounded domains take a lower
bound, an upper bound and an initial value. The initial value must belong to
the active domain; bounds are not widened automatically.

Omitted parameters retain their configuration and AutoML behavior. For a
parameter without a domain, a plain value preserves the component's original
behavior; it does not invent a search range. ``Const`` also works without a
domain. Its value is validated against the component's constraints, regardless
of any retained, inactive search bounds.

A constant can lie outside a retained inactive domain. Reactivating that
domain with a plain value requires the new initial value to lie within it.
See :ref:`pipeline-guide-parameters` for the fixed-to-optimizable transition.

For ``Selection.configure``, plain values preserve each variant's own domain;
an explicit domain replaces it on every target. Any incompatible target
rejects the whole operation. An empty selection changes nothing.

Validation occurs as soon as the domain is known and no later than candidate
generation. Configuration attached to an open family is checked when that
family resolves, before any of its candidates is trained.

.. _pipelines-ref-analysis:

Analytical collections
----------------------

``metrics``, ``statistics`` and ``explanations`` build collections of
computations. They support naming, addition, removal, replacement, cloning and
selection. ``configure`` targets individual components, directly or through
``find_all``. These collections have no ``start`` policy and do not multiply
candidate pipelines.

With positional arguments, a collection constructor declares exactly the
supplied analyses. With no arguments, it uses the default collection, which
you can then edit by addition or subtraction. An explicit empty list in
``IAML(statistics=[], explanations=[])`` disables those collective
computations. Omitting configuration retains the defaults.

Analytical parameters configure fixed computations. For example,
``pos_label=1``, ``k=10`` and ``nsamples=100`` are never explored by the
optimizer. ``Const`` is accepted but redundant for these parameters.
``Int`` and ``Float`` domains are rejected in analytical collections.

All applicable operations run when the corresponding computation is
requested. Applicability depends on the dataset, target, model and capabilities
required by the method. An inapplicable operation is reported with available
reasons; its absence is not converted into a zero-valued result.

.. _pipelines-ref-iaml:

The IAML entry point
--------------------

The constructor accepts pipeline and analysis recipes alongside its existing
search and execution options:

.. code-block:: text

   IAML(
       pipeline=...,
       metrics=...,
       statistics=...,
       explanations=...,
       main_metric=...,
       ...,
   )

Omitting ``pipeline`` uses the built-in AutoML preset, including its minimal
predictor strategy. An explicit ``IAML(pipeline=recipe)`` permits only the
recipe's branches; no extra minimal predictors are added implicitly.
``PipelineSpec.default()`` describes the same preset as ``IAML()``.

The ``main`` and ``minimal`` strategies are visible and editable. The
``predictor`` group belongs to ``main``; ``minimal_predictor`` belongs to
``minimal``. Removing ``minimal`` removes its candidates. Its descendants
have aliases distinct from those of ``main``, so shortcuts such as
``pipeline.normalize`` and ``pipeline.predictor`` address the main strategy.

``search.pipeline``, ``search.metrics``, ``search.statistics`` and
``search.explanations`` expose the study's attached, editable recipes.
Construction copies the supplied recipes. A launch compiles another copy
of those definitions into new execution instances. Subsequent edits prepare
a future launch, without modifying trained candidates or the ranking of a
search already underway.

``main_metric`` accepts a ``Metric`` instance or a metric alias in the
configured collection. An instance retains its parameters; resolving the
collection does not reset them to defaults. If several variants make
instance-based selection ambiguous, use a precise alias. An unknown or
removed main metric raises an error and is not silently reintroduced.

Metric aliases propagate to scores, including separate variants of the same
class. Variants whose output keys cannot be distinguished are rejected,
rather than silently overwriting results. The objective's definition and
ranking direction remain stable throughout the search. A candidate whose
main metric is inapplicable cannot be ranked.

.. _pipelines-ref-computations:

Computation methods and results
-------------------------------

.. list-table:: Study and model methods
   :header-rows: 1
   :widths: 36 64

   * - Call
     - Behavior and result
   * - ``search.fit(X, y)``
     - Returns trained models/candidates. Metrics are computed for validation
       and ranking.
   * - ``search.get_descriptive_statistics(X, y)``
     - Describes the supplied data without requiring ``fit``. Returns
       descriptive results without replacing a remembered training dataset.
   * - ``search.get_descriptive_statistics()``
     - Describes the last raw search dataset, possibly sampled, before pipeline
       transformations. Returns an empty table if no dataset is remembered.
   * - ``search.visualize_descriptive_statistics()``
     - Produces figures for the descriptive statistics of the search dataset.
   * - ``model.describe_metrics()``
     - Presents internal validation scores with their distinct identities.
   * - ``model.evaluate(X_test, y_test)``
     - Computes scores on supplied data using the model's configured metrics.
       Does not overwrite remembered validation scores.
   * - ``model.explain(X, y=None)``
     - Runs applicable explanations and plots in its collection. Returns
       results keyed by alias or the component's unambiguous default key.
   * - ``model.explain_feature_importance(X, nsamples=...)``
     - Runs the specialized feature-importance method and returns an
       ``Explanation``.

Explicit descriptive requests require both ``X`` and ``y`` because some
statistics require a target. Descriptive caches account for the analyzed data
and statistical configuration. Returned tables are independent copies,
including mutable cell contents. Intermediate preprocessing states are not
automatically described.

Explanations run on demand on a fitted model. Performance plots requiring a
target are skipped when it is absent. Expensive analyses do not automatically
run for every validation candidate. Each model retains a copy of its analysis
collections; subsequent edits to ``search.explanations`` do not modify it.

``KernelSHAP`` adapts IAML's Kernel SHAP mechanism to a recipe for classification
and regression. The adapter reports survival tasks as inapplicable. SHAP is
imported only when the computation runs. ``nsamples`` controls computational effort per
prediction, rather than the number of observations explained. The explanation
uses the second probability column when available, otherwise ``predict``.
Selecting another class, TreeSHAP and LIME are not supported by this adapter.

.. _pipelines-ref-inspection:

Inspection and reconstruction
-----------------------------

``pipeline.describe()`` presents the declared tree, allowed and initial
alternatives, families, exclusions and additions, followed by parameter state:
active domain, initial or constant value and any retained inactive domain.
An open family is marked unresolved before launch.

``describe`` and ``diff`` return a ``RecipeReport`` implementing ``Mapping``.
``print(report)`` produces readable output; ``report["tree"]`` accesses a
description's content and ``report.to_dict()`` returns an independent copy.
A diff exposes ``changes``, ``before`` and ``after``. ``to_code`` returns a
Python code string.

``pipeline.diff(base)`` describes recipe differences from a base: blocks,
alternatives, start policies and parameters. It does not compare training
results.

``pipeline.to_code()`` generates code to reconstruct the declaration. It
preserves aliases, domains, constants, inactive domains and explicit policies.
Reconstruction may require several calls. Custom component classes are
referenced by import; their implementations are not copied.

After ``fit``, ``search.describe()`` presents the recipe actually compiled,
resolved collections and families, ranking metric, evaluated candidates and
observed applicability decisions with available reasons. Static inspection
cannot predict every ``suitable()`` decision that depends on earlier
transformations.

The launch report retains the recipe version and resolved catalogue for that
search. Code for an open family reconstructs that family declaration; it
does not guarantee an identical catalogue in another environment. Recipe
reconstruction alone does not guarantee identical numerical results.

``search.describe()`` is also a ``RecipeReport``. Its status is ``declared``
before launch and ``resolved`` for the most recent launch. Later edits do not
rewrite the report of that launch.

.. _pipelines-ref-validation:

Validation and diagnostics
--------------------------

Diagnostics identify the affected object or alias and the condition preventing
computation. ``TypeError`` indicates an incompatible form or component family,
``KeyError`` an unknown alias, ``AttributeError`` an unknown component parameter
and ``ValueError`` an invalid configuration or edit.

.. list-table:: Validation conditions
   :header-rows: 1

   * - Condition
     - Diagnostic
   * - Duplicate explicit alias, unknown alias or class passed to ``[]``.
     - Name conflict, missing target or unsupported selector.
   * - Positional alternatives and ``tag`` supplied together to ``choice``.
     - Mutually exclusive construction modes.
   * - Unknown parameter, incorrect type or value violating a component
       constraint.
     - Affected parameter, component and constraint.
   * - Initial value outside the domain or inconsistent bounds.
     - Affected value and bounds, without implicit correction.
   * - Collective configuration incompatible with a target.
     - Incompatible target; none of the targets is modified.
   * - Search domain used in an analysis.
     - Analytical parameters are fixed; the domain is rejected.
   * - Replacement with a different root alias or an analytical replacement
       from a different family.
     - Incompatible replacement identity or family.
   * - Partial start without component-replacement support in the optimizer.
     - Search mode incompatible with the declared scope.
   * - Unknown, excluded or ambiguous main metric; indistinguishable score keys.
     - Objective or variants requiring an explicit selection.
   * - Candidate path without exactly one terminal predictor.
     - Structurally invalid branch, rejected before training.

A component inapplicable to data is a runtime applicability decision, distinct
from a malformed recipe. Reports list skipped operations and available reasons.
Automatic metric filters may express a recommendation: an explicitly requested
objective, by instance or alias, is retained despite such a filter. A missing,
invalid or uncomputable main score on any fold rejects the candidate.

Collective analyses report ``success``, ``inapplicable`` or ``error`` for each
result key. A failed applicable calculation is reported with its reason while
successful results from other methods are retained. Configuration errors are
rejected before computation. Descriptive reports are available through
``search.describe()["analysis_reports"]["statistics"]``. Candidates expose
``metric_report`` for validation, ``evaluation_report`` for external evaluation
and ``explanation_report`` for requested explanations.

An aggregated secondary score requires complete validation coverage.
``model.metric_coverage`` retains available values, their count and the total
number of folds. Incomplete coverage produces no aggregated score. Explicitly
named statistics use their aliases in table rows, with ``alias : label`` when
a statistic produces multiple rows.

.. _pipelines-ref-open:

Supported scope and limitations
-------------------------------

The API supports bounded ``Int`` and ``Float`` domains and retains existing
categorical component configurations. The following features are outside its
supported scope:

* Atomic replacement of whole subpipelines during partial exploration.
* Logarithmic distributions, general domains and additional conditional
  constraints.
* A public task filter argument in ``predictors``.
* Descriptive snapshots between every transformation and TreeSHAP or LIME
  explanation adapters.

Existing JSON interfaces remain available alongside ``to_code`` reconstruction.
Classes and callable values used by generated code must be importable: local
classes and lambdas cannot be exported by this method.
