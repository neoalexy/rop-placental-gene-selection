import random

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def fitness(X, y, cols, cv, penalty=0.001):
    if len(cols) < 2:
        return 0.0
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    auc = cross_val_score(rf, X[:, cols], y, cv=cv, scoring="roc_auc").mean()
    return auc - penalty * len(cols)


class GA:
    """Binary-chromosome genetic algorithm for gene selection.

    Fitness is the cross-validated Random Forest AUC minus a small penalty
    per selected gene. The RNG state lives on the instance, so calling `run`
    several times (e.g. once per outer fold) continues the same stream.
    """

    def __init__(self, pop=30, gens=30, mut=0.02, k=3, folds=5, min_genes=5, max_genes=20,
                 seed=42, n_jobs=2):
        self.pop, self.gens, self.mut, self.k = pop, gens, mut, k
        self.folds, self.min_genes, self.max_genes = folds, min_genes, max_genes
        self.n_jobs = n_jobs
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def run(self, X, y):
        """X is a samples x genes DataFrame. Returns the selected gene names and fitness history."""
        genes, Xv = list(X.columns), X.to_numpy()
        n = len(genes)
        cv = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=42)
        cache = {}

        def evaluate(population):
            keys = [tuple(np.flatnonzero(c)) for c in population]
            todo = list(dict.fromkeys(k for k in keys if k not in cache))
            scores = Parallel(n_jobs=self.n_jobs)(delayed(fitness)(Xv, y, list(k), cv) for k in todo)
            cache.update(zip(todo, scores))
            return [cache[k] for k in keys]

        population = [self._random(n) for _ in range(self.pop)]
        history = []
        for _ in range(self.gens):
            scores = evaluate(population)
            best = int(np.argmax(scores))
            history.append(scores[best])
            nxt = [population[best].copy()]
            while len(nxt) < self.pop:
                child = self._crossover(self._select(population, scores), self._select(population, scores))
                nxt.append(self._mutate(child))
            population = nxt

        scores = evaluate(population)
        best = population[int(np.argmax(scores))]
        return [genes[i] for i in np.flatnonzero(best)], history

    def _random(self, n):
        c = np.zeros(n, dtype=int)
        size = self.rng.randint(self.min_genes, min(self.max_genes, n))
        c[self.rng.sample(range(n), size)] = 1
        return c

    def _select(self, population, scores):
        idx = self.rng.sample(range(len(population)), self.k)
        return population[max(idx, key=lambda i: scores[i])].copy()

    def _crossover(self, a, b):
        return np.where(self.np_rng.randint(0, 2, size=len(a)) == 1, a, b)

    def _mutate(self, c):
        for i in range(len(c)):
            if self.rng.random() < self.mut:
                c[i] = 1 - c[i]
        return c
