import numpy as np
from scipy.optimize import minimize


class LogisticRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def sigmoid(self, x):
        return 1 / (1.0 + np.exp(-x))

    def objective(self, beta, X, y):
        n, _ = X.shape
        tmp = np.dot(X, beta[1:]) + beta[0]
        object = np.sum(np.log(1+np.exp(tmp))) - np.dot(y, tmp)

        return object / (1.0 * n)

    def gradient(self, beta, X, y):
        n, _ = X.shape
        tmp = self.sigmoid(np.dot(X, beta[1:]) + beta[0]) - y
        grads = []
        grads.append(np.sum(tmp)/(1.0*n))
        grads.extend(np.dot(X.T, tmp)/(1.0*n))

        return np.array(grads)

    def callback(self, beta):
        self.history.append(self.objective(beta, self.X, self.y))

    def fit(self, X, y):
        n, p = X.shape
        self.X = X
        self.y = y
        self.history = []

        # initialize
        beta0 = np.random.randn(p+1)

        res = minimize(self.objective, beta0, args=(X, y), method='BFGS',
                       jac=self.gradient, callback=self.callback)

        self.coef = res.x[1:]
        self.intercept = res.x[0]

        return self

    def predict(self, X):
        activation = self.sigmoid(np.dot(X, self.coef) + self.intercept)
        y = np.zeros(X.shape[0], dtype=np.int)
        y[activation >= 0.5] = 1

        return y

    def score(self, X, y):
        y_predict = self.predict(X)

        return np.mean(y == y_predict) * 100

    @property
    def coef_(self):
        return self.coef

    @property
    def intercept_(self):
        return self.intercept

    @property
    def history_(self):
        return self.history
