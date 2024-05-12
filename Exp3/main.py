from sklearn import svm, tree
import numpy as np
import argparse
import os
from utils import get_data
from models import Baseline, Bagging, AdaBoost
import sys

# arg parser for command line arguments
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--regressor', type=str, default='svm')
    parser.add_argument('--vectorizer', type=str, default='tfidf')
    parser.add_argument('--ensemble', type=str, default='bagging')
    parser.add_argument('--max_depth', type=int, default=20)
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()

    X_train, y_train, X_test, y_test = get_data() # split data and transform text data to vectors

    # select regressor
    if args.regressor == 'svm':
        regressor = svm.LinearSVR(random_state=0)
    elif args.regressor == 'tree':
        regressor = tree.DecisionTreeRegressor(max_depth=args.max_depth, random_state=0)
    else:
        raise ValueError('Invalid regressor: {}'.format(args.regressor))

    # select ensemble
    if args.ensemble == 'bagging':
        ensemble = Bagging(args.n, args.ratio, regressor)
    elif args.ensemble == 'adaboost':
        ensemble = AdaBoost(args.n, regressor)
    elif args.ensemble == 'baseline':
        ensemble = Baseline(regressor)
    else:
        raise ValueError('Invalid ensemble: {}'.format(args.ensemble))

    ensemble.fit(X_train, y_train) # fit the ensemble model using the training data
    y_pred = ensemble.predict(X_test) # predict the test data

    # calculate metrics
    mae = np.mean(np.abs(y_pred - y_test))
    mse = np.mean((y_pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    print("MAE: %.4f, MSE: %.4f, RMSE: %.4f" % (mae, mse, rmse))
