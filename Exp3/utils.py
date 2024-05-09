import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import os
import random
import pickle
import numpy as np

def get_vectorizer(vectorizer_type='count'):
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
    # df['reviewerID'] = df['reviewerID'].astype(str)
    # df['asin'] = df['asin'].astype(str)
    # df['summary'] = df['summary'].astype(str)
    # df['reviewText'] = df['reviewText'].astype(str)
    # matrix = vectorizer.fit_transform(list(df['reviewerID'] + ' ' + df['asin'] + ' ' + df['summary'] + ' ' + df['reviewText']))
    X = vectorizer.transform(list(df['reviewText'])).toarray()
    # df['overall'] = df['overall'].astype(int)
    y = list(df['overall'])
    # return X, y
    return np.array(X), np.array(y)

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
        vectorizer = get_vectorizer(vectorizer_type)
        vectorizer = vectorizer.fit(list(train['reviewText']))
        with open('vectorizer_%s.pkl' % vectorizer_type, 'wb') as f:
            pickle.dump(vectorizer, f)

    print("Processing data...")
    X_train, y_train = process_data(train, vectorizer)
    X_test, y_test = process_data(test, vectorizer)

    return X_train, y_train, X_test, y_test



    # if os.path.exists('train_%s.csv' % vectorizer_type) and os.path.exists('test_%s.csv' % vectorizer_type):
    #     train_data = pd.read_csv('train_%s.csv' % vectorizer_type)
    #     test_data = pd.read_csv('test_%s.csv' % vectorizer_type)

    #     X_train = train_data.drop('overall', axis=1).values
    #     y_train = train_data['overall'].values
    #     X_test = test_data.drop('overall', axis=1).values
    #     y_test = test_data['overall'].values
    # else:
    #     random.seed(0)
    #     train, test = split_data(df)
    #     X_train, y_train = process_data(train)
    #     X_test, y_test = process_data(test)

    #     train_data = pd.DataFrame(X_train)
    #     train_data['overall'] = y_train
    #     train_data.to_csv('train_%s.csv' % vectorizer_type, index=False)

    #     test_data = pd.DataFrame(X_test)
    #     test_data['overall'] = y_test
    #     test_data.to_csv('test_%s.csv' % vectorizer_type, index=False)

    # return X_train, y_train, X_test, y_test
