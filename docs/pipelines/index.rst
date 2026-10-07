.. _pipelines:

============
Pipeline API
============

Use the pipeline API when your team needs to adapt the AutoML preset, assemble
reusable strategies or integrate its own components. Recipes describe the
allowed pipelines and their parameter domains; IAML handles candidate
generation, cross-validation and final fitting with the same engine as the
:doc:`integrated workflow <../quick_start>`.

If you are new to this part of IAML, start with :doc:`../discover_pipelines`
for a visual tour, or the :doc:`example catalogue <../examples/pipelines/README>`
for runnable Python examples. Pipeline configuration remains optional for
training with the defaults.

Choose your next step
=====================

.. list-table:: Advanced guides
   :header-rows: 1
   :widths: 40 60

   * - Goal
     - Read
   * - Adapt the preset or build your own pipeline.
     - :doc:`guide`: composition, subtraction, editing, reusable fragments
       and parameter domains.
   * - Configure the calculations and search around it.
     - :doc:`study`: objectives, validation settings, descriptive statistics,
       explanations and analytical reports.
   * - Find a method's exact behavior.
     - :doc:`reference`: signatures, naming rules, atomic edits and diagnostics.
   * - Run a complete example.
     - :doc:`../examples/pipelines/README`: thirteen runnable Python
       examples.
   * - Implement a component for your team.
     - :doc:`../adaptability`: transformation, prediction, analysis and other
       extension contracts.

A recipe and a fitted model have different roles. ``IAML(pipeline=recipe)``
copies the declaration; ``fit`` creates execution instances. Edit the attached
copy through ``search.pipeline`` to prepare another run. Trained ``Candidate``
objects retain their own fitted ``IAMLPipeline`` and analysis definitions.

.. toctree::
   :maxdepth: 1

   guide
   study
   reference
   Examples <../examples/pipelines/README>
