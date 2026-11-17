"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


MODEL_CONFIGS = {
    "random_forest_full": (RandomForestClassifier, {"n_estimators": 300, "min_samples_leaf": 1, "n_jobs": 1}),
    "random_forest_regularized": (RandomForestClassifier, {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 3, "n_jobs": 1}),
    "extra_trees_full": (ExtraTreesClassifier, {"n_estimators": 300, "min_samples_leaf": 1, "n_jobs": 1}),
    "extra_trees_regularized": (ExtraTreesClassifier, {"n_estimators": 300, "min_samples_leaf": 3, "n_jobs": 1}),
    "gradient_boosting": (GradientBoostingClassifier, {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2}),
    "hist_gradient_boosting_7": (HistGradientBoostingClassifier, {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 7, "l2_regularization": 1.0, "early_stopping": False}),
    "hist_gradient_boosting_15": (HistGradientBoostingClassifier, {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0, "early_stopping": False}),
}
SELECTED_MODEL = "random_forest_regularized"


def make_model(name, random_seed):
    model_type, parameters = MODEL_CONFIGS[name]
    return model_type(random_state=random_seed, **parameters)


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    model = make_model(SELECTED_MODEL, data.random_seed)
    model.fit(np.asarray(data.train_features, dtype=float), np.asarray(data.train_target).reshape(-1))
    features = np.asarray(data.test_features, dtype=float)
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(features)[:, positive_index]
    predictions = model.predict(features)
    return EnsembleResult(
        model_name=str(model),
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(data.test_target, probabilities)),
        accuracy=float(accuracy_score(data.test_target, predictions)),
        precision=float(precision_score(data.test_target, predictions, zero_division=0)),
        recall=float(recall_score(data.test_target, predictions, zero_division=0)),
        f1=float(f1_score(data.test_target, predictions, zero_division=0)),
    )
