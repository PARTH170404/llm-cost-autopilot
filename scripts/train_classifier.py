"""
scripts/train_classifier.py
Trains complexity classifier on labeled dataset and saves model artifacts.
Run: python scripts/train_classifier.py
"""

import sys
import csv
import pickle
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from app.classifier.features import extract_features
from app.utils.config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

DATASET_PATH = "data/prompts_dataset.csv"
MODEL_PATH = "data/classifier.pkl"
VECTORIZER_PATH = "data/vectorizer.pkl"
LABEL_ENCODER_PATH = "data/label_encoder.pkl"
MIN_ACCURACY = 0.80


def load_dataset():
    prompts, labels = [], []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row["prompt"].strip())
            labels.append(row["complexity"].strip())
    return prompts, labels


def build_features(prompts):
    return np.array([extract_features(p) for p in prompts])


def train():
    logger.info("Loading dataset...")
    prompts, labels = load_dataset()
    logger.info(f"Loaded {len(prompts)} samples: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    logger.info(f"Classes: {le.classes_}")

    # Build feature matrix
    X = build_features(prompts)
    logger.info(f"Feature matrix shape: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train multiple models, pick best
    models = {
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

    best_model = None
    best_score = 0.0
    best_name = ""

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        mean_cv = cv_scores.mean()
        logger.info(f"{name}: CV accuracy = {mean_cv:.3f} ± {cv_scores.std():.3f}")
        if mean_cv > best_score:
            best_score = mean_cv
            best_model = model
            best_name = name

    logger.info(f"\nBest model: {best_name} (CV: {best_score:.3f})")

    # Train best model on full training set
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"  CLASSIFIER TRAINING RESULTS")
    print("=" * 60)
    print(f"  Model         : {best_name}")
    print(f"  CV Accuracy   : {best_score:.3f}")
    print(f"  Test Accuracy : {test_accuracy:.3f}")
    print(f"  Min Required  : {MIN_ACCURACY:.3f}")
    print()
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("=" * 60)

    if test_accuracy < MIN_ACCURACY:
        logger.warning(f"Accuracy {test_accuracy:.3f} below minimum {MIN_ACCURACY}. Consider adding more training data.")

    # Save artifacts
    Path("data").mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info(f"Label encoder saved to {LABEL_ENCODER_PATH}")

    if test_accuracy >= MIN_ACCURACY:
        print(f"\n✅ PASS: Accuracy {test_accuracy:.3f} meets threshold {MIN_ACCURACY}")
    else:
        print(f"\n⚠️  WARNING: Accuracy {test_accuracy:.3f} below threshold {MIN_ACCURACY}")

    return test_accuracy >= MIN_ACCURACY


if __name__ == "__main__":
    success = train()
    sys.exit(0 if success else 1)
