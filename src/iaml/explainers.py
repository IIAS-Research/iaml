"""Explanation components used by the declarative study API."""
from __future__ import annotations

class KernelSHAP:
    """Explain a fitted pipeline with its existing Kernel SHAP implementation.

    ``nsamples`` controls the effort for each explained row. SHAP is imported
    only when this component is computed, through ``IAMLPipeline.explain_model``.
    """

    _flow_kind = "explanations"

    def __init__(self, nsamples: int = 20) -> None:
        if isinstance(nsamples, bool) or not isinstance(nsamples, int) or nsamples <= 0:
            raise ValueError("nsamples must be a positive integer")
        self.nsamples = nsamples

    def __str__(self) -> str:
        return "kernel_shap"

    def suitable(self, dataset) -> bool:
        return dataset.needed_estimator in {"classifier", "regressor"}

    def compute(self, pipeline, X, y=None, **kwargs):
        """Return the existing ``Explanation`` object for the supplied rows."""
        return pipeline.explain_model(X, nsamples=self.nsamples)


__all__ = ["KernelSHAP"]
