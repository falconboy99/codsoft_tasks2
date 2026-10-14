"""Spam SMS detection: TF-IDF features + Naive Bayes / Logistic Regression / Linear SVM.

Usage:
    python train.py                      # train, evaluate, save best model
    python train.py --predict "WIN a FREE prize, text now!"
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "spam.csv"  # Kaggle SMS Spam Collection (columns v1=label, v2=text)
OUT = ROOT / "results"
MODEL = ROOT / "model" / "spam_classifier.joblib"
SEED = 42


def features():
    # Word unigrams+bigrams plus character n-grams (robust to "fr33", "txt", URLs, numbers)
    return FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])


CANDIDATES = {
    "naive_bayes": (MultinomialNB(), {"clf__alpha": [0.01, 0.05, 0.1, 0.3, 1.0]}),
    "logistic_regression": (LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
                            {"clf__C": [1, 10, 30, 100]}),
    "linear_svm": (LinearSVC(class_weight="balanced"), {"clf__C": [0.1, 0.3, 1, 3]}),
}


def load():
    df = pd.read_csv(DATA, encoding="latin-1", usecols=["v1", "v2"]).rename(columns={"v1": "label", "v2": "text"})
    df["y"] = (df["label"] == "spam").astype(int)
    return df


def metrics(y, p):
    return {"accuracy": accuracy_score(y, p), "precision": precision_score(y, p),
            "recall": recall_score(y, p), "f1": f1_score(y, p)}


def train():
    df = load()
    Xtr, Xte, ytr, yte = train_test_split(df["text"], df["y"], test_size=0.2, stratify=df["y"], random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    OUT.mkdir(exist_ok=True)
    results, fitted = {}, {}
    for name, (clf, grid) in CANDIDATES.items():
        gs = GridSearchCV(Pipeline([("tfidf", features()), ("clf", clf)]), grid, scoring="f1", cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr)
        pred = gs.predict(Xte)
        m = metrics(yte, pred)
        m.update(cv_f1=gs.best_score_, best_params=gs.best_params_,
                 confusion_matrix=confusion_matrix(yte, pred).tolist())
        results[name], fitted[name] = m, gs.best_estimator_
        print(f"\n== {name} {gs.best_params_}  CV-F1={gs.best_score_:.4f}")
        print(classification_report(yte, pred, target_names=["ham", "spam"], digits=4))

    # Select on cross-validated F1 (train only) so the test set stays an unbiased estimate
    best = max(results, key=lambda k: results[k]["cv_f1"])
    model = fitted[best]
    if best == "linear_svm":  # add probabilities for predict_proba
        model = CalibratedClassifierCV(model, cv=5).fit(Xtr, ytr)
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "name": best, "labels": ["ham", "spam"]}, MODEL)

    test_df = pd.DataFrame({"text": Xte, "true": yte, "pred": fitted[best].predict(Xte)})
    test_df[test_df.true != test_df.pred].assign(
        true=lambda d: d.true.map({0: "ham", 1: "spam"}), pred=lambda d: d.pred.map({0: "ham", 1: "spam"})
    ).to_csv(OUT / "misclassified.csv", index=False)

    table = pd.DataFrame({k: {c: v[c] for c in ["accuracy", "precision", "recall", "f1", "cv_f1"]}
                          for k, v in results.items()}).T.round(4)
    print("\n", table, f"\n\nBest model: {best} -> {MODEL}")
    table.to_csv(OUT / "comparison.csv")
    json.dump({"best_model": best, "n_train": len(Xtr), "n_test": len(Xte), "models": results},
              open(OUT / "metrics.json", "w"), indent=2, default=str)


def predict(texts):
    bundle = joblib.load(MODEL)
    m = bundle["model"]
    probs = m.predict_proba(texts)[:, 1]
    for t, p in zip(texts, probs):
        print(f"{'SPAM' if p >= 0.5 else 'HAM ':4}  p(spam)={p:.3f}  {t}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", nargs="+")
    a = ap.parse_args()
    predict(a.predict) if a.predict else train()
