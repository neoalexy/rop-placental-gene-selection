import numpy as np
import pandas as pd
import pytest

from src.data import load, log_cpm
from src.deg import KNOWN, pool
from src.ga import GA
from src.validation import OUTER


@pytest.fixture(scope="module")
def data():
    expr, y, _ = load()
    return expr, y


def test_cpm_sums_to_a_million(data):
    expr, _ = data
    assert np.allclose((2 ** expr - 1).sum(axis=1), 1e6)


def test_pool_matches_thesis(data):
    expr, y = data
    genes = pool(expr, y)
    assert len(genes) == 124
    assert set(KNOWN.values()) <= set(genes)
    assert genes == [g for g in expr.columns if g in set(genes)]  # matrix order, not p-value order


def test_fold_pools_use_training_samples_only(data):
    expr, y = data
    sizes = [len(pool(expr.iloc[tr], y[tr])) for tr, _ in OUTER.split(expr, y)]
    assert sizes == [227, 118, 93, 127, 111]


def test_ga_reports_the_columns_it_used():
    rng = np.random.RandomState(0)
    y = np.array([0, 1] * 15)
    X = pd.DataFrame(rng.normal(size=(30, 12)), columns=[f"noise{i}" for i in range(12)])
    X["signal"] = y * 3 + rng.normal(scale=0.3, size=30)
    X = X.sample(frac=1, axis=1, random_state=1)  # put the signal somewhere in the middle

    picked, _ = GA(pop=8, gens=4, folds=3, max_genes=6, n_jobs=1, seed=0).run(X, y)
    again, _ = GA(pop=8, gens=4, folds=3, max_genes=6, n_jobs=1, seed=0).run(X, y)

    assert "signal" in picked
    assert set(picked) <= set(X.columns)
    assert picked == again
