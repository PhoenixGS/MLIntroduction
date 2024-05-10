import pickle
import os
import numpy as np
import copy
from sklearn.utils import resample
from sklearn import svm, tree
from utils import calc_weighted_median

class Bagging:
    def __init__(self, n, ratio, regressor):
        self.n = n
        self.ratio = ratio
        self.regressor = regressor

    def fit(self, X, y):
        if os.path.exists('regressors_%s_%s_%s_%s.pkl' % (self, self.n, self.ratio, self.regressor)):
            print('Loading regressors...')
            with open('regressors_%s_%s_%s_%s.pkl' % (self, self.n, self.ratio, self.regressor), 'rb') as f:
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
            with open('regressors_%s_%s_%s_%s.pkl' % (self, self.n, self.ratio, self.regressor), 'wb') as f:
                pickle.dump(self.regressors, f)

    def predict(self, X):
        print('Predicting...')
        y_pred = np.zeros(X.shape[0])
        for regressor in self.regressors:
            y_pred += regressor.predict(X)
        y_pred /= len(self.regressors)
        return y_pred

class AdaBoost:
    def __init__(self, n, regressor):
        self.n = n
        self.regressor = regressor

    # regressive model
    def fit(self, X, y):
        if os.path.exists('regressors_%s_%s_%s.pkl' % (self.__class__.__name__, self.n, self.regressor.__class__.__name__)) and os.path.exists('alphas_%s_%s_%s.pkl' % (self.__class__.__name__, self.n, self.regressor.__class__.__name__)):
            print('Loading regressors...')
            with open('regressors_%s_%s_%s.pkl' % (self.__class__.__name__, self.n, self.regressor.__class__.__name__), 'rb') as f:
                self.regressors = pickle.load(f)
            with open('alphas_%s_%s_%s.pkl' % (self.__class__.__name__, self.n, self.regressor.__class__.__name__), 'rb') as f:
                self.alphas = pickle.load(f)
        else:
            print('Fitting regressors...')
            self.regressors = []
            self.alphas = []
            w = np.ones(len(y)) / len(y)
            for i in range(self.n):
                print("Fitting regressor %d" % i)
                regressor = copy.deepcopy(self.regressor)
                regressor.fit(X, y, sample_weight=w)
                y_pred = regressor.predict(X)
                Em = np.max(np.abs(y_pred - y))
                emi = np.abs(y_pred - y) / Em
                em = np.sum(w * emi)
                alpha = 0.5 * np.log((1 - em) / em)
                w = w * np.exp(1 - emi) / np.sum(w * np.exp(1 - emi))
                assert(np.abs(np.sum(w) - 1) < 1e-6)
                self.regressors.append(regressor)
                self.alphas.append(alpha)
            with open('regressors_%s_%s_%s.pkl' % (self.__class__.__name__, self.n, self.regressor.__class__.__name__), 'wb') as f:
                pickle.dump(self.regressors, f)
            with open('alphas_%s_%s_%s.pkl' % (self.__class__.__name__, self.n, self.regressor.__class__.__name__), 'wb') as f:
                pickle.dump(self.alphas, f)

    def predict(self, X):
        print('Predicting...')
        y_pred = []
        for regressor, alpha in zip(self.regressors, self.alphas):
            y_pred.append(regressor.predict(X))
        y_pred = np.array(y_pred)
        y_pred = y_pred.T
        y_res = []
        for i in range(len(y_pred)):
            y_res.append(calc_weighted_median(y_pred[i], np.log(1 / np.array(self.alphas))))
        return np.array(y_res)
