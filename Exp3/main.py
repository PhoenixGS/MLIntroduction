import csv
from sklearn import svm, tree
import numpy as np
import argparse
import os
from utils import get_data
from models import Bagging, AdaBoost
import sys

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

    X_train, y_train, X_test, y_test = get_data()
    # train, test, vectorizer = get_data(args.vectorizer)
    # X_train = train['reviewText']
    # y_train = train['overall']
    # X_test = test['reviewText']
    # y_test = test['overall']

    if args.regressor == 'svm':
        regressor = svm.LinearSVR()
    elif args.regressor == 'tree':
        regressor = tree.DecisionTreeRegressor(max_depth=args.max_depth)

    if args.ensemble == 'bagging':
        ensemble = Bagging(args.n, args.ratio, regressor)
    elif args.ensemble == 'adaboost':
        ensemble = AdaBoost(args.n, regressor)

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)

    mse = np.mean((y_pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    print("MSE: %.4f, RMSE: %.4f" % (mse, rmse))

    print(y_pred[:10])
    print(y_test[:10])
