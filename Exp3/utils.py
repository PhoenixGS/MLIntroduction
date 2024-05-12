import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import os
import random
import pickle
import numpy as np

def select_vectorizer(vectorizer_type='count'):
    if vectorizer_type == 'count':
        return CountVectorizer()
    elif vectorizer_type == 'tfidf':
        return TfidfVectorizer()
    else:
        raise ValueError('Invalid vectorizer type: {}'.format(vectorizer_type))

def split_data(df, ratio=0.9):
    train = df.sample(frac=ratio)
    test = df.drop(train.index)
    return train, test

def process_data(df, vectorizer):
    X = vectorizer.transform(list(df['reviewText']))
    y = list(df['overall'])
    return X, y

def get_data(vectorizer_type='count'):
    print("Reading data...")
    df = pd.read_csv('exp3-reviews.csv', delimiter='\t')
    
    random.seed(0) 
    train, test = split_data(df)

    print("Getting vectorizer...")
    if os.path.exists('vectorizer_%s.pkl' % vectorizer_type):
        with open('vectorizer_%s.pkl' % vectorizer_type, 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        vectorizer = select_vectorizer(vectorizer_type)
        vectorizer = vectorizer.fit(list(train['reviewText']))
        with open('vectorizer_%s.pkl' % vectorizer_type, 'wb') as f:
            pickle.dump(vectorizer, f)
    
    # return train, test, vectorizer
    print("Processing data...")
    X_train, y_train = process_data(train, vectorizer)
    X_test, y_test = process_data(test, vectorizer)

    return X_train, y_train, X_test, y_test

def calc_weighted_median(y, w):
    assert(len(y) == len(w))
    w = w / np.sum(w)
    sorted_idx = np.argsort(y)
    y_sorted = y[sorted_idx]
    w_sorted = w[sorted_idx]
    cumsum = np.cumsum(w_sorted)
    median_idx = np.searchsorted(cumsum, 0.5)
    return y_sorted[median_idx]
