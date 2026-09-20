"""A small classification workflow with performance plots, SHAP and an audit."""

import json
import random
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from iaml import IAML, RocAucMetric


def main():
    random.seed(42)
    np.random.seed(42)
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X = X.iloc[:, :6].sample(n=120, random_state=42)
    y = y.loc[X.index].eq(0).astype(int)  # 1 = malignant, 0 = benign.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    search = IAML(
        max_duration=30, max_workers=1,
        main_metric=RocAucMetric(), keep_training_history=True,
    )
    search.fit(X_train, y_train)
    chosen_model = search.chosen_candidate
    test_scores = chosen_model.evaluate(X_test, y_test)
    print(chosen_model.describe_steps())
    print(test_scores)

    output = Path("iaml_example")
    output.mkdir(exist_ok=True)
    for plot in chosen_model.explain_model_performance(X_test, y_test):
        (output / f"{type(plot).__name__}.png").write_bytes(plot.image)

    explanation = chosen_model.explain_feature_importance(X_test.iloc[:5], nsamples=64)
    for kind in ("bar", "waterfall"):
        (output / f"shap-{kind}.png").write_bytes(explanation.to_plot(kind).image)

    report = {
        "pipeline": chosen_model.pipeline_audit_summary(),
        "test_scores": test_scores,
        "training_history": search.training_history,
    }
    (output / "analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
