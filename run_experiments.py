import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ensemble_models import MODEL_CONFIGS, SELECTED_MODEL, make_model, run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput


ROOT = Path(__file__).resolve().parent
SEED = 42
NUMERIC = [
    "Duration_of_Credit_monthly", "Credit_Amount", "Instalment_per_cent",
    "Duration_in_Current_address", "Age_years", "No_of_Credits_at_this_Bank",
    "No_of_dependents",
]


def load_split():
    frame = pd.read_csv(ROOT / "german.csv", sep=";")
    return train_test_split(frame.iloc[:, 1:], frame.iloc[:, 0], test_size=0.2,
                            random_state=SEED, stratify=frame.iloc[:, 0])


def make_preprocessor(columns):
    categorical = [name for name in columns if name not in NUMERIC]
    return ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


def run_cross_validation(train_x, train_y):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "roc_auc": "roc_auc", "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    experiments = []
    for name in MODEL_CONFIGS:
        pipeline = Pipeline([
            ("prepare", make_preprocessor(train_x.columns)),
            ("model", make_model(name, SEED)),
        ])
        scores = cross_validate(pipeline, train_x, train_y, cv=folds,
                                scoring=scoring, n_jobs=1, error_score="raise")
        experiments.append({
            "name": name,
            "model": type(pipeline.named_steps["model"]).__name__,
            "parameters": MODEL_CONFIGS[name][1],
            "cv_mean": {metric: float(scores[f"test_{metric}"].mean()) for metric in scoring},
            "cv_std": {metric: float(scores[f"test_{metric}"].std()) for metric in scoring},
            "cv_folds": {metric: scores[f"test_{metric}"].tolist() for metric in scoring},
        })
    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-only", action="store_true")
    arguments = parser.parse_args()
    train_x, test_x, train_y, test_y = load_split()
    experiments = run_cross_validation(train_x, train_y)
    selected = max(experiments, key=lambda item: item["cv_mean"]["roc_auc"])["name"]
    results = {
        "dataset": "german.csv",
        "dataset_sha256": hashlib.sha256((ROOT / "german.csv").read_bytes()).hexdigest(),
        "random_seed": SEED,
        "split": {"train_size": len(train_x), "test_size": len(test_x), "stratified": True,
                  "train_indices_sha256": hashlib.sha256(np.asarray(train_x.index, dtype="<i8").tobytes()).hexdigest(),
                  "test_indices_sha256": hashlib.sha256(np.asarray(test_x.index, dtype="<i8").tobytes()).hexdigest()},
        "selection": {"criterion": "maximum mean ROC-AUC on 5-fold stratified CV within train",
                      "selected": selected, "cv_shuffle": True},
        "experiments": experiments,
    }
    if arguments.cv_only:
        print(json.dumps(results, indent=2))
        return
    if selected != SELECTED_MODEL:
        raise ValueError(f"CV winner {selected} does not match the fixed model {SELECTED_MODEL}")
    prepare = make_preprocessor(train_x.columns)
    train_features = prepare.fit_transform(train_x)
    test_features = prepare.transform(test_x)
    result = run_ensemble_experiment(EnsembleInput(train_features, train_y, test_features, test_y, SEED))
    summary = asdict(result)
    summary.pop("probabilities")
    summary.pop("predictions")
    results["feature_count"] = train_features.shape[1]
    results["test"] = summary
    results["test"]["confusion_matrix"] = {
        "tn": int(((test_y.to_numpy() == 0) & (result.predictions == 0)).sum()),
        "fp": int(((test_y.to_numpy() == 0) & (result.predictions == 1)).sum()),
        "fn": int(((test_y.to_numpy() == 1) & (result.predictions == 0)).sum()),
        "tp": int(((test_y.to_numpy() == 1) & (result.predictions == 1)).sum()),
    }
    (ROOT / "experiment_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
