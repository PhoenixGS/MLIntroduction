import pickle
import os
import numpy as np
import copy
from sklearn.utils import resample

class Bagging:
    def __init__(self, n, ratio, regressor):
        self.n = n
        self.ratio = ratio
        self.regressor = regressor

    def fit(self, X, y):
        if os.path.exists('regressors.pkl'):
            print('Loading regressors...')
            with open('regressors.pkl', 'rb') as f:
                self.regressors = pickle.load(f)
        else:
            print('Fitting regressors...')
            self.regressors = []
            for i in range(self.n):
                # idx = np.random.choice(len(y), int(self.ratio * len(y)), replace=True)
                # X_sample = X[idx]
                # y_sample = y[idx]
                print(X.shape, len(y))
                X_sample, y_sample = resample(X, y, n_samples=int(self.ratio * len(y)), replace=True)
                print(X_sample.shape, len(y_sample))
                regressor = copy.deepcopy(self.regressor)
                print("Fitting regressor %d" % i)
                regressor.fit(X_sample, y_sample)
                self.regressors.append(regressor)
            with open('regressors.pkl', 'wb') as f:
                pickle.dump(self.regressors, f)

    def predict(self, X):
        print('Predicting...')
        y_pred = np.zeros(X.shape[0])
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
