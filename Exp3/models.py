import pickle
import os

class Bagging:
    def __init__(self, n, ratio, regressor):
        self.n = n
        self.ratio = ratio
        self.regressor = regressor

    def fit(self, X, y):
        if os.path.exists('regressors.pkl'):
            with open('regressors.pkl', 'rb') as f:
                self.regressors = pickle.load(f)
        else:
            self.regressors = []
            for i in range(self.n):
                idx = np.random.choice(len(X), int(self.ratio * len(X)), replace=True)
                X_sample = X[idx]
                y_sample = y[idx]
                regressor = copy.deepcopy(self.regressor)
                regressor.fit(X_sample, y_sample)
                self.regressors.append(regressor)
            with open('regressors.pkl', 'wb') as f:
                pickle.dump(self.regressors, f)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for regressor in self.regressors:
            y_pred += regressor.predict(X)
        y_pred /= len(self.regressors)
        return y_pred

class AdaBoost:
    def __init__(self, n, ratio, regressor):
        self.n = n
        self.ratio = ratio
        self.regressor = regressor

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
