.. iaml documentation master file, created by
   sphinx-quickstart on Thu Nov 28 10:20:08 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========================================
IAML: A Modular and Explainable AutoML
========================================

Welcome to the documentation of **IAML** (Incremental AutoML), a high-performance, modular, and explainable AutoML framework designed to address the limitations of existing solutions for tabular data.

Introduction
=============
IAML (Incremental AutoML) is a modular and adaptable framework designed to simplify the creation of optimized and explainable machine learning pipelines. Its architecture allows seamless customization, enabling users to tailor preprocessing steps, model selection, and evaluation metrics to suit diverse datasets and domains.

IAML employs a genetic algorithm-inspired optimization process, iteratively refining pipelines through strategies like mutation and selection. This approach ensures efficient exploration and improvement of model configurations. With built-in tools for transparency, such as SHAP and Yellowbrick visualizations, IAML balances performance, adaptability, and explainability, making it an effective solution for automated machine learning.

Key Features
============
- **Simplicity:** Develop a complete machine learning solution effortlessly with minimal code.
- **Ease of Use:** A Python API inspired by Scikit-learn for effortless integration.
- **Explainability:** Provides a suite of tools, including SHAP-based insights and Yellowbrick visualizations, to enhance model transparency and understanding.
- **Modularity:** Flexible architecture that can be adapted to domain-specific needs.
- **Efficiency:** Inspired by genetic algorithms, its optimization approach ensures performante pipeline tuning.

Basic example
=============
Here is an example to get started with IAML:

.. code-block:: python

   from iaml import IAML

   if __name__ == "__main__":
       search = IAML(max_duration=30, max_workers=1)
       candidates = search.fit(X_train, y_train)
       predictions = candidates[0].predict(X_test)

See :doc:`quick_start` for a complete example with pandas DataFrames.
For more advanced usage, such as custom metrics or explainability features, refer to the :doc:`usage` section.

Documentation Structure
========================
- **Introduction:** Overview of IAML's motivation and features.
- **Quick Start:** Step-by-step guide to start your IAML journey.
- **Architecture:** Explanation of the modular design and optimization strategy.
- **IAML for science:** Learn how IAML can help you :doc:`create scientific knowledge<scientific>`.
- **Explainability:** Fully understand your IAML pipeline in the :doc:`explainability` documentation.
- **Adaptability:** Refer to :doc:`adaptability` to dive deeper into IAML's architecture and learn how to add your own code.
- **References:** Detailed citations and additional resources.

How to Cite IAML
=================

If you use IAML in your work, please consider citing it. Below is the recommended citation format:

.. code-block::
   
   Merieux, R., Ruellet, H., Bourachot, R., Dahlouk, Y. A., Blanchard, F., Vuiblet, V. 
   "IAML: A Modular and Explainable AutoML Framework for High-Performance on Tabular Data."
   Preprint submitted to Knowledge-Based Systems, 2024.


Contribute to IAML
===================
IAML is an open-source project hosted on GitHub: `IAML GitHub Repository <https://github.com/iias-research/iaml>`_.

We welcome contributions, bug reports, and feature requests from the community.

---

Start exploring IAML today and take your machine learning projects to the next level!
