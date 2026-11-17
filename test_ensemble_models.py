import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ensemble_models import run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput
from run_experiments import load_split, make_preprocessor


ROOT = Path(__file__).resolve().parent


def manual_metrics(target, probabilities, predictions):
    target = np.asarray(target).reshape(-1)
    positives = probabilities[target == 1]
    negatives = probabilities[target == 0]
    auc = sum(float(p > n) + 0.5 * float(p == n) for p in positives for n in negatives)
    auc /= len(positives) * len(negatives)
    tp = int(((target == 1) & (predictions == 1)).sum())
    fp = int(((target == 0) & (predictions == 1)).sum())
    fn = int(((target == 1) & (predictions == 0)).sum())
    return {
        "roc_auc": auc,
        "accuracy": float((target == predictions).mean()),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


class EnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(123)
        features = rng.normal(size=(160, 5))
        target = (features[:, 0] + features[:, 1] ** 2 + rng.normal(size=160) * 0.4 > 0.7).astype(int)
        train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.25,
                                                         stratify=target, random_state=42)
        cls.data = EnsembleInput(train_x, train_y, test_x, test_y)
        cls.result = run_ensemble_experiment(cls.data)

    def test_probabilities_and_predictions_have_correct_shape(self):
        result = self.result
        self.assertIn("RandomForestClassifier", result.model_name)
        self.assertEqual(result.probabilities.shape, (40,))
        self.assertEqual(result.predictions.shape, (40,))
        self.assertTrue(np.isfinite(result.probabilities).all())
        self.assertTrue(((result.probabilities >= 0) & (result.probabilities <= 1)).all())
        self.assertTrue(set(result.predictions).issubset({0, 1}))
        np.testing.assert_array_equal(result.predictions, (result.probabilities > 0.5).astype(int))

    def test_all_metrics_match_independent_formulas(self):
        expected = manual_metrics(self.data.test_target, self.result.probabilities, self.result.predictions)
        for name, value in expected.items():
            self.assertAlmostEqual(getattr(self.result, name), value, places=12)

    def test_test_targets_cannot_change_the_fitted_model(self):
        changed = EnsembleInput(self.data.train_features, self.data.train_target,
                                self.data.test_features, 1 - self.data.test_target)
        result = run_ensemble_experiment(changed)
        np.testing.assert_array_equal(result.probabilities, self.result.probabilities)
        np.testing.assert_array_equal(result.predictions, self.result.predictions)
        self.assertAlmostEqual(result.roc_auc, 1 - self.result.roc_auc)

    def test_fixed_seed_reproduces_probabilities(self):
        result = run_ensemble_experiment(self.data)
        np.testing.assert_array_equal(result.probabilities, self.result.probabilities)

    def test_numeric_dataframes_and_column_targets_are_supported(self):
        data = EnsembleInput(pd.DataFrame(self.data.train_features),
                             self.data.train_target.reshape(-1, 1),
                             pd.DataFrame(self.data.test_features), self.data.test_target)
        result = run_ensemble_experiment(data)
        np.testing.assert_array_equal(result.probabilities, self.result.probabilities)

    def test_unseen_category_is_not_fitted_from_test(self):
        train_x, test_x, _, _ = load_split()
        prepare = make_preprocessor(train_x.columns)
        train_features = prepare.fit_transform(train_x)
        changed = test_x.copy()
        changed["Purpose"] = 999999
        test_features = prepare.transform(changed)
        self.assertEqual(train_features.shape[1], test_features.shape[1])
        encoder = prepare.named_transformers_["categorical"]
        self.assertFalse(any(999999 in categories for categories in encoder.categories_))
        self.assertTrue(np.isfinite(test_features).all())

    def test_saved_metrics_are_reproduced_on_german_credit(self):
        train_x, test_x, train_y, test_y = load_split()
        prepare = make_preprocessor(train_x.columns)
        result = run_ensemble_experiment(EnsembleInput(prepare.fit_transform(train_x), train_y,
                                                      prepare.transform(test_x), test_y))
        stored = json.loads((ROOT / "experiment_results.json").read_text())["test"]
        expected = manual_metrics(test_y, result.probabilities, result.predictions)
        for name, value in expected.items():
            self.assertAlmostEqual(stored[name], value, places=12)
            self.assertAlmostEqual(getattr(result, name), value, places=12)


if __name__ == "__main__":
    unittest.main()
