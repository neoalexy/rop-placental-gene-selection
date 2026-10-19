import json

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from .data import RESULTS
from .deg import pool
from .ga import GA

OUTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
INNER = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)


def cached(name, fn):
    """Run fn() once and keep the JSON result in results/, so notebooks re-run in seconds."""
    path = RESULTS / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    out = fn()
    path.write_text(json.dumps(out, indent=2))
    return out


def nested_ga(expr, y, known=True, seed=42, log=print):
    """Nested CV: DEG filtering and the whole GA run only on the 24 training samples of each outer fold."""
    ga = GA(folds=4, seed=seed)
    folds = []
    for i, (tr, te) in enumerate(OUTER.split(expr, y), 1):
        genes = pool(expr.iloc[tr], y[tr], known=known)
        chosen, _ = ga.run(expr.iloc[tr][genes], y[tr])
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(expr.iloc[tr][chosen], y[tr])
        auc = roc_auc_score(y[te], rf.predict_proba(expr.iloc[te][chosen])[:, 1])
        folds.append({"fold": i, "auc": auc, "pool": len(genes), "genes": chosen})
        log(f"fold {i}: AUC={auc:.3f}  pool={len(genes)}  selected={len(chosen)}")
    return folds


def repeated_nested_ga(expr, y, seeds):
    """Nested GA repeated with different GA seeds (same outer folds), one cache file per seed."""
    runs = {}
    for s in seeds:
        folds = cached(f"nested_ga_seed{s}", lambda: nested_ga(expr, y, seed=s, log=lambda *_: None))
        runs[s] = [f["auc"] for f in folds]
    return runs


def _regression(penalty, grid):
    solver = {"l1": "liblinear", "l2": "lbfgs", "elasticnet": "saga"}[penalty]
    model = LogisticRegression(penalty=penalty, solver=solver, max_iter=5000)
    return GridSearchCV(model, grid, cv=INNER, scoring="roc_auc")


BASELINES = {
    "LASSO (L1)": lambda: _regression("l1", {"C": [0.001, 0.01, 0.1, 1, 10]}),
    "Ridge (L2)": lambda: _regression("l2", {"C": [0.001, 0.01, 0.1, 1, 10]}),
    "Elastic-net": lambda: _regression("elasticnet", {"C": [0.01, 0.1, 1, 10], "l1_ratio": [0.1, 0.5, 0.9]}),
}


def nested_baselines(expr, y):
    """Regularized logistic regression on the same per-fold DEG pool as the GA."""
    out = {name: [] for name in BASELINES}
    for tr, te in OUTER.split(expr, y):
        genes = pool(expr.iloc[tr], y[tr])
        Xtr, Xte = expr.iloc[tr][genes], expr.iloc[te][genes]
        for name, make in BASELINES.items():
            model = make().fit(Xtr, y[tr])
            auc = roc_auc_score(y[te], model.predict_proba(Xte)[:, 1])
            nonzero = int(np.sum(model.best_estimator_.coef_ != 0))
            out[name].append({"auc": auc, "features": nonzero})
    return out


def rf_baseline(expr, y, k=124):
    """Random Forest on the top-k genes by F-test, with selection refit inside each fold."""
    model = make_pipeline(SelectKBest(f_classif, k=k), RandomForestClassifier(n_estimators=200, random_state=42))
    return cross_val_score(model, expr, y, cv=OUTER, scoring="roc_auc").tolist()


def permutation_test(X, y, n=500, seed=42):
    """CV AUC of a fixed gene set on real vs shuffled labels."""
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    real = cross_val_score(rf, X, y, cv=OUTER, scoring="roc_auc").mean()
    rng = np.random.RandomState(seed)
    shuffled = [rng.permutation(y) for _ in range(n)]
    null = Parallel(n_jobs=2)(delayed(_cv_auc)(rf, X, ys) for ys in shuffled)
    return {"real": real, "null": null}


def _cv_auc(model, X, y):
    return cross_val_score(model, X, y, cv=OUTER, scoring="roc_auc").mean()
