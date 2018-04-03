"""
Reference:
    http://jocelynchi.com/a-coordinate-descent-algorithm-for-the-lasso-problem
"""

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import r2_score


class Lasso:
    """
    The optimization objective of Lasso is
        1/(2 * n) * ||y - Xw||^2_2 + alpha * ||w||_1
    """

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=1000, tol=0.0001, selection='cyclic'):
        self._alpha = alpha
        self._fit_intercept = fit_intercept
        self._normalize = normalize
        self._copy_X = copy_X
        self._max_iter = max_iter
        self._tol = tol
        self._selection = selection

        if normalize:
            self._scaler = preprocessing.StandardScaler()

    def compute_step(self, k, X, y, coef, intercept, alpha):
        n, p = X.shape
        y_predict = np.dot(X, coef) + intercept
        rk = np.dot(X[:, k], y - y_predict + X[:, k] * coef[k])
        rk = rk / (1.0 * n)
        zk = np.linalg.norm(X[:, k], ord=2) ** 2
        zk = zk / (1.0 * n)
        coef_k = (np.amax([rk-alpha, 0]) - np.amax([-rk-alpha, 0]))
        coef_k = coef_k / (1.0 * zk)

        return coef_k

    def objective(self, X, y, coef, intercept, alpha):
        n, p = X.shape
        total = 0

        y_predict = np.dot(X, coef) + intercept
        total += \
            1/(2.0*n) * np.linalg.norm(y-y_predict, ord=2) ** 2
        total += alpha * np.linalg.norm(coef, ord=1)

        return total

    def fit(self, X, y):
        if self._copy_X:
            X = X.copy()
        if self._normalize:
            X = self._scaler.fit_transform(X)
        self._objectives = []

        # initialize data
        num_samples, num_features = X.shape
        coef = np.zeros(num_features)
        old_coef = np.zeros(num_features)
        intercept = 0
        if self._fit_intercept:
            tmp = y - np.dot(X, coef)
            intercept = np.sum(tmp) / (1.0 * num_samples)
        num_iters = 0
        for iter in range(self._max_iter):
            num_iters = num_iters + 1
            if (self._selection == 'cyclic'):
                for k in range(num_features):
                    old_coef[k] = coef[k]
                    coef[k] = self.compute_step(k, X, y, coef,
                                                intercept, self._alpha)
                if self._fit_intercept:
                    tmp = y - np.dot(X, coef)
                    intercept = np.sum(tmp) / (1.0 * num_samples)

                # check condition of convergence
                coef_updates = np.abs(coef - old_coef)
                if np.amax(coef_updates) < self._tol:
                    break
            self._objectives.append(self.objective(X, y, coef,
                                                   intercept, self._alpha))

        self._coef = coef
        self._intercept = intercept
        self._num_iters = num_iters

        return self

    def predict(self, X):
        if self._copy_X:
            X = X.copy()
        if self._normalize:
            X = self._scaler.transform(X)

        y_predict = np.dot(X, self._coef) + self._intercept

        return y_predict

    def score(self, X, y):
        y_predict = self.predict(X)

        return r2_score(y, y_predict)

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept

    @property
    def n_iter_(self):
        return self._num_iters

    @property
    def objectives_(self):
        return self._objectives

    def __str__(self):
        return ('Lasso(alpha={}, copy_X={}, '
                'fit_intercept={}, max_iter={}, '
                'normalize={}, selection=\'{}\', '
                'tol={})').format(self._alpha, self._copy_X,
                                  self._fit_intercept, self._max_iter,
                                  self._normalize, self._selection,
                                  self._tol)


if __name__ == '__main__':
    pass
