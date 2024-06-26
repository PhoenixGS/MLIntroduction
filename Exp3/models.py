import pickle
import os
import numpy as np
import copy
from sklearn.utils import resample
from sklearn import svm, tree
from utils import calc_weighted_median

class Baseline:
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        if os.path.exists('regressor_%s_%s.pkl' % (self.__class__.__name__, self.regressor.__class__.__name__)): # if the regressor has been saved, load it
            print('Loading regressor...')
            with open('regressor_%s_%s.pkl' % (self.__class__.__name__, self.regressor.__class__.__name__), 'rb') as f:
                self.regressor = pickle.load(f)
        else:
            print('Fitting regressor...')
            self.regressor.fit(X, y) # fit the regressor
            with open('regressor_%s_%s.pkl' % (self.__class__.__name__, self.regressor.__class__.__name__), 'wb') as f:
                pickle.dump(self.regressor, f)

    def predict(self, X):
        print('Predicting...')
        return self.regressor.predict(X)

class Bagging:
    def __init__(self, n, ratio, regressor):
        self.n = n
        self.ratio = ratio
        self.regressor = regressor

    def fit(self, X, y):
        if os.path.exists('regressors_%s_%s_%s_%s.pkl' % (self.__class__.__name__, self.n, self.ratio, self.regressor.__class__.__name__)): # if the regressors have been saved, load them
            print('Loading regressors...')
            with open('regressors_%s_%s_%s_%s.pkl' % (self.__class__.__name__, self.n, self.ratio, self.regressor.__class__.__name__), 'rb') as f:
                self.regressors = pickle.load(f)
        else:
            print('Fitting regressors...')
            self.regressors = [] # regressors is empty at the beginning
            for i in range(self.n):
                X_sample, y_sample = resample(X, y, n_samples=int(self.ratio * len(y)), replace=True, random_state=i) # sample data for fitting
                regressor = copy.deepcopy(self.regressor)
                print("Fitting regressor %d" % i)
                regressor.fit(X_sample, y_sample)
                self.regressors.append(regressor)
            with open('regressors_%s_%s_%s_%s.pkl' % (self.__class__.__name__, self.n, self.ratio, self.regressor.__class__.__name__), 'wb') as f:
                pickle.dump(self.regressors, f)

    def predict(self, X):
        print('Predicting...')
        y_pred = np.zeros(X.shape[0])
        for regressor in self.regressors: # predict using each regressor
            y_pred += regressor.predict(X)
        y_pred /= len(self.regressors) # average the predictions
        return y_pred

class AdaBoost:
    def __init__(self, n, regressor):
        self.n = n
        self.regressor = regressor

    # regressive model
    def fit(self, X, y):
        if os.path.exists('regressors_%s_%s_%s.pkl' % (self.__class__.__name__, self.n, self.regressor.__class__.__name__)) and os.path.exists('alphas_%s_%s_%s.pkl' % (self.__class__.__name__, self.n, self.regressor.__class__.__name__)): # if the regressors and alphas have been saved, load them
            print('Loading regressors...')
            with open('regressors_%s_%s_%s.pkl' % (self.__class__.__name__, self.n, self.regressor.__class__.__name__), 'rb') as f:
                self.regressors = pickle.load(f)
            with open('alphas_%s_%s_%s.pkl' % (self.__class__.__name__, self.n, self.regressor.__class__.__name__), 'rb') as f:
                self.alphas = pickle.load(f)
        else:
            print('Fitting regressors...')
            self.regressors = []
            self.alphas = []
            w = np.ones(len(y)) / len(y) # initialize weights
            for i in range(self.n):
                print("Fitting regressor %d" % i)
                regressor = copy.deepcopy(self.regressor)
                regressor.fit(X, y, sample_weight=w * len(y)) # fit the regressor with weights
                y_pred = regressor.predict(X)
                Em = np.max(np.abs(y_pred - y)) # calculate the maximum error
                emi = np.abs(y_pred - y) / Em
                em = np.sum(w * emi)
                alpha = (1 - em) / em # calculate the weight of the regressor
                w = w * np.exp(1 - emi) / np.sum(w * np.exp(1 - emi)) # update the weights
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
            y_pred.append(regressor.predict(X)) # predict using each regressor
        y_pred = np.array(y_pred)
        y_pred = y_pred.T
        y_res = []
        for i in range(len(y_pred)):
            y_res.append(calc_weighted_median(y_pred[i], np.log(1 / np.array(self.alphas)))) # calculate the weighted median
        return np.array(y_res)
