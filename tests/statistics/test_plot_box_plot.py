"""Unit tests for BoxPlot."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import unittest

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def _load_boxplot_class():
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "src"
    plots_path = src_path / "iaml" / "plots"
    boxplot_path = plots_path / "box_plot.py"

    _ensure_package("iaml", src_path / "iaml")
    _ensure_package("iaml.plots", plots_path)

    module_name = "iaml.plots.box_plot"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, boxplot_path)
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load BoxPlot module.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.BoxPlot


BoxPlot = _load_boxplot_class()


class TestBoxPlot(unittest.TestCase):
    """Coverage for BoxPlot core behavior."""

    def _assert_plot_image(self, plot) -> None:
        image = plot.image
        self.assertIsInstance(image, (bytes, bytearray))
        self.assertGreater(len(image), 0)

    def _ensure_bxp(self) -> bool:
        if hasattr(plt, "bxp"):
            return False

        def _bxp_stub(*_args, **_kwargs):
            return []

        plt.bxp = _bxp_stub
        return True

    def test_numeric_box_plot(self) -> None:
        stats_df = pd.DataFrame(
            {
                "num_1_all": [0.1, 1.1, 0.25, 0.5, 0.75],
                "num_1_class0": [0.2, 1.0, 0.3, 0.55, 0.8],
            },
            index=["min", "max", "quantile_0.25", "quantile_0.5", "quantile_0.75"],
        )

        stubbed_bxp = self._ensure_bxp()
        try:
            plot = BoxPlot().compute(stats_df, base_name="num_1")
        finally:
            if stubbed_bxp and hasattr(plt, "bxp"):
                delattr(plt, "bxp")

        self._assert_plot_image(plot)

    def test_missing_required_rows_placeholder(self) -> None:
        stats_df = pd.DataFrame({"num_1_all": [1.0]}, index=["min"])

        plot = BoxPlot().compute(stats_df, base_name="num_1")

        self._assert_plot_image(plot)

    def test_empty_dataframe_placeholder(self) -> None:
        plot = BoxPlot().compute(pd.DataFrame())

        self._assert_plot_image(plot)
