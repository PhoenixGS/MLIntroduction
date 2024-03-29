import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
from string import punctuation as punct
from tqdm import tqdm
from math import log
import random

alpha = 1
beta = 15
trans = str.maketrans({key: " " for key in punct})

# check number and garbled code
def check(s):
    return not any(char.isdigit() for char in s) and str.isascii(s)

# check feature
def important(s):
    lis = ["received", "subject", "sender", "from"]
    return any(voca in s for voca in lis)


class NaiveBayes:
    def __init__(self):
        self.dict = [{}, {}]
        self.cnt = [0, 0]

    def train(self, x_train, y_train):
        print("Training...")
        for (x, y) in tqdm(zip(x_train, y_train)):
            for lines in x.lower().split("\n"):
                # check feature
                if important(lines):
                    c = beta
                else:
                    c = 1

                # count
                for voca in lines.translate(trans).lower().split():
                    if check(voca):
                        if not voca in self.dict[y].keys():
                            self.dict[y][voca] = 0
                        self.dict[y][voca] += c
            self.cnt[y] += 1

        # remove vocabulary which only counts one
        self.dict[0] = {key: value for (key, value) in self.dict[0].items() if value > 1}
        self.dict[1] = {key: value for (key, value) in self.dict[1].items() if value > 1}
        print("Dictionary length: ", len(self.dict[0]), len(self.dict[1]))
        
    def predict(self, X):
        print("Predicting...")
        sum_value = [sum(self.dict[0].values()), sum(self.dict[1].values())]
        len_dict = [len(self.dict[0]), len(self.dict[1])]
        sss = [log(self.cnt[0] / sum(self.cnt)), log(self.cnt[1] / sum(self.cnt))]
        y = []
        for x in tqdm(X):
            ss = sss.copy()
            for lines in x.lower().split("\n"):
                # check feature
                if important(lines):
                    c = beta
                else:
                    c = 1
                    
                for voca in lines.translate(trans).lower().split():
                    if check(voca):
                        # calculate probability
                        if voca in self.dict[0].keys():
                            ss[0] += c * log((self.dict[0][voca] + alpha) / (sum_value[0] + alpha * len_dict[0]))
                        else:
                            ss[0] += c * log(alpha / (sum_value[0] + alpha * len_dict[0]))
                        if voca in self.dict[1].keys():
                            ss[1] += c * log((self.dict[1][voca] + alpha) / (sum_value[1] + alpha * len_dict[1]))
                        else:
                            ss[1] += c * log(alpha / (sum_value[1] + alpha * len_dict[1]))
            if ss[0] > ss[1]:
                y.append(0)
            else:
                y.append(1)
        return y

if __name__ == "__main__":
    with open("./trec06p/label/index", "r") as file:
        data = file.readlines()

    # read data
    data = [line.strip().split(' ') for line in data]
    data = [(label, os.path.join("trec06p", path.replace('../', ''))) for [label, path] in data]

    print("Reading data...")
    x = []
    for [label, path] in tqdm(data):
        with open(path, "r", encoding='utf-8', errors='ignore') as file:
            x.append(file.read())
    
    y = [label for [label, path] in data]

    # set random seed
    random.seed(42)
    
    for ratio in [0.05, 0.5, 1]:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        accu = []
        prec = []
        reca = []
        f1 = []
        
        # 5-fold cross-validation
        for fold_, (train_index, test_index) in enumerate(folds.split(x, y)):
            model = NaiveBayes()
            
            ratio_train_index = random.sample((list)(train_index), (int)(ratio * len(train_index)))
            x_train = [x[i] for i in ratio_train_index]
            y_train = [y[i] == "ham" for i in ratio_train_index]
            x_test = [x[i] for i in test_index]
            y_test = [y[i] == "ham" for i in test_index]

            model.train(x_train, y_train)
            yh_test = model.predict(x_test)
            
            yh_test = np.array(yh_test)
            y_test = np.array(y_test)
            
            # calculate indicators
            accuracy = np.mean(yh_test == y_test)
            accu.append(accuracy)

            TP = np.sum((yh_test == 1) & (y_test == 1))
            TN = np.sum((yh_test == 0) & (y_test == 0))
            FP = np.sum((yh_test == 1) & (y_test == 0))
            FN = np.sum((yh_test == 0) & (y_test == 1))
            
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1 = 2 * TP / (2 * TP + FP + FN)
            prec.append(precision)
            reca.append(recall)
            f1.append(F1)

        print("Average score of ratio ", ratio, ": Accuracy: ", np.mean(accu), "Precision: ", np.mean(prec), "Recall: ", np.mean(reca), "F1: ", np.mean(f1))
