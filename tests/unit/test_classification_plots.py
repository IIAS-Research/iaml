"""Performance curves must describe the fitted probability column consistently."""
import unittest
import warnings
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.flow import explanations, metrics, use
from iaml.iaml_pipeline import IAMLPipeline
from iaml.logger import Logger
from iaml.metrics import RocAucMetric
from iaml.plots import PrecisionRecallCurvePlot, ROCAUCPlot
from iaml.study_analyses import compile_analyses, compile_metrics


class TestClassificationPlots(unittest.TestCase):
    def setUp(self):
        verbose = Logger().verbose
        Logger().verbose = 0
        self.addCleanup(setattr, Logger(), "verbose", verbose)
        self.addCleanup(plt.close, "all")

    @staticmethod
    def make_candidate(negative="healthy", positive="sick"):
        X = pd.DataFrame({"score": [0.1, 0.6, 0.4, 0.9]}, index=[10, 30, 20, 40])
        y = pd.Series([negative, negative, positive, positive], index=X.index)
        train_X = pd.concat([X] * 3, ignore_index=True)
        train_y = pd.concat([y] * 3, ignore_index=True)
        estimator = LogisticRegression(random_state=0).fit(train_X, train_y)
        dataset = Dataset(train_X, train_y)
        metric_definitions, objective, _ = compile_metrics(
            metrics(use(RocAucMetric).named("auc")), "auc", dataset,
        )
        candidate = Candidate(
            dataset, main_metric=objective, metric_definitions=metric_definitions,
            explanation_definitions=compile_analyses(explanations(
                use(ROCAUCPlot).named("roc"),
                use(PrecisionRecallCurvePlot).named("pr"),
            )),
            iaml_pipeline=IAMLPipeline([("model", estimator)], estimator_type="classifier"),
        )
        return candidate, X, y

    def compute_curves(self, candidate, X, y):
        with patch.object(plt, "plot", wraps=plt.plot) as plot:
            results = candidate.explain(X, y)
        self.assertEqual(set(results), {"roc", "pr"}, candidate.explanation_report)
        self.assertTrue(all(record["status"] == "success"
                            for record in candidate.explanation_report))
        for result in results.values():
            self.assertTrue(result.image.startswith(b"\x89PNG"))
        return [call for call in plot.call_args_list
                if call.kwargs.get("label", "").startswith(("ROC curve", "Precision-Recall curve"))]

    def test_curve_values_match_metrics_and_are_invariant_to_row_order(self):
        for negative, positive in ((0, 1), (-1, 1), (False, True), (1, 2), (2, 3),
                                   ("healthy", "sick")):
            candidate, X, y = self.make_candidate(negative, positive)
            baseline = None
            for order in ([0, 1, 2, 3], [2, 3, 0, 1], [3, 1, 2, 0]):
                with self.subTest(labels=(negative, positive), order=order):
                    frame, targets = X.iloc[order], y.iloc[order]
                    auc_score = candidate.evaluate(frame, targets)["auc"]
                    probabilities = candidate.predict_proba(frame)[:, 1]
                    ap_score = average_precision_score(
                        np.asarray(targets) == candidate.pipeline.classes_[1], probabilities,
                    )
                    self.assertAlmostEqual(auc_score, 0.75)
                    self.assertAlmostEqual(ap_score, 5 / 6)

                    roc, pr = self.compute_curves(candidate, frame, targets)
                    fpr, tpr = map(np.asarray, roc.args[:2])
                    recall, precision = map(np.asarray, pr.args[:2])
                    self.assertAlmostEqual(float(np.trapz(tpr, fpr)), auc_score)
                    plotted_ap = float(-np.sum(np.diff(recall) * precision[:-1]))
                    self.assertAlmostEqual(plotted_ap, ap_score)
                    self.assertIn(f"AUC = {auc_score:.2f}", roc.kwargs["label"])
                    self.assertIn(f"AP = {ap_score:.2f}", pr.kwargs["label"])
                    if baseline is None:
                        baseline = (fpr, tpr, recall, precision)
                    else:
                        for expected, observed in zip(baseline, (fpr, tpr, recall, precision)):
                            np.testing.assert_allclose(observed, expected)

    def test_lists_arrays_and_single_column_frames_are_supported(self):
        candidate, X, y = self.make_candidate()
        for targets in (y.tolist(), y.to_numpy(), y.to_frame("target")):
            with self.subTest(container=type(targets).__name__):
                roc, pr = self.compute_curves(candidate, X, targets)
                self.assertIn("AUC = 0.75", roc.kwargs["label"])
                self.assertIn("AP = 0.83", pr.kwargs["label"])

    def test_missing_test_class_does_not_change_the_positive_class(self):
        candidate, X, _ = self.make_candidate()
        for label, expected_ap in (("healthy", 0.0), ("sick", 1.0)):
            with self.subTest(label=label):
                with warnings.catch_warnings(), patch.object(plt, "plot", wraps=plt.plot) as plot:
                    warnings.simplefilter("ignore", UserWarning)
                    result = candidate.explain(X, [label] * len(X))
                self.assertEqual(set(result), {"pr"})
                report = {record["key"]: record for record in candidate.explanation_report}
                self.assertEqual(report["roc"]["status"], "error")
                self.assertIn("both fitted classes", report["roc"]["reason"])
                self.assertEqual(report["pr"]["status"], "success")
                pr = next(call for call in plot.call_args_list
                          if call.kwargs.get("label", "").startswith("Precision-Recall curve"))
                recall, precision = map(np.asarray, pr.args[:2])
                plotted_ap = float(-np.sum(np.diff(recall) * precision[:-1]))
                self.assertAlmostEqual(plotted_ap, expected_ap)

    def test_unknown_test_labels_are_reported(self):
        candidate, X, y = self.make_candidate()
        y.iloc[0] = "unknown"
        self.assertEqual(candidate.explain(X, y), {})
        self.assertTrue(all(record["status"] == "error" and
                            "absent from the fitted classes" in record["reason"]
                            for record in candidate.explanation_report))

    def test_multiclass_estimators_have_an_explicit_binary_constraint(self):
        X = pd.DataFrame({"score": np.arange(12, dtype=float)})
        y = pd.Series(["healthy", "ill", "recovered"] * 4)
        estimator = LogisticRegression(random_state=0).fit(X, y)
        definitions = compile_analyses(explanations(ROCAUCPlot, PrecisionRecallCurvePlot))
        candidate = Candidate(
            Dataset(X, y), explanation_definitions=definitions,
            iaml_pipeline=IAMLPipeline([("model", estimator)], estimator_type="classifier"),
        )
        self.assertEqual(candidate.explain(X, y), {})
        self.assertTrue(all(record["status"] == "inapplicable"
                            for record in candidate.explanation_report))
        for plot_class in (ROCAUCPlot, PrecisionRecallCurvePlot):
            with self.subTest(plot=plot_class.__name__):
                self.assertFalse(plot_class.suitable("multiclass"))
                with self.assertRaisesRegex(ValueError, "exactly two classes"):
                    plot_class().compute(estimator, X, y)


if __name__ == "__main__":
    unittest.main()
