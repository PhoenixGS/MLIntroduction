import torch
import numpy as np
import random
from torchvision import datasets

def load_data():
    train_data = datasets.MNIST(root = './data', train=True, download=True)
    images = train_data.data.numpy()
    labels = train_data.targets.numpy()
    return images, labels

def pr(image):
    for i in range(len(image)):
        for j in range(len(image[i])):
            print('.' if image[i][j] < 128 else '#', end='')
        print()

class k_means:
    def __init__(self, data, label=None, k=10):
        self.data = data
        # self.label = label
        self.k = k
        # self.centers = np.array([self.data[np.argwhere(label == i)[0][0]] for i in range(k)])
        # self.centers = np.array([data[i] for i in random.sample(range(len(data)), k)])
        index = np.array(random.sample(range(len(data)), k))
        self.centers = data[index]
        self.label = np.zeros(len(data))
        self.update_label()

    def update_label(self):
        for i in range(len(self.data)):
            distances = [np.linalg.norm(self.data[i] - self.centers[c]) for c in range(self.k)]
            cluster = np.argmin(distances)
            self.label[i] = cluster
    
    def iterate(self):
        for c in range(self.k):
            #self.centers[c] = np.mean(self.data[np.argwhere(self.label == c)], axis=0)
            if np.sum(self.label == c) == 0:
                assert(False)
            index = np.argwhere(self.label == c)
            index = index.reshape(index.shape[0])
            center = np.mean(self.data[index], axis=0)
            #self.centers[c] = self.data[np.argmin([np.linalg.norm(center - self.data[i]) for i in range(len(self.data))])]
            self.centers[c] = center
        self.update_label()

    def max_distance(self):
        maxdistance = 0
        for i in range(len(self.data)):
            distance = np.linalg.norm(self.data[i] - self.centers[int(self.label[i])])
            if distance > maxdistance:
                maxdistance = distance
        return maxdistance

    def calc_label(self, labels):
        self.caption = np.zeros(self.k)
        for k in range(self.k):
            label = np.argmax([np.sum(labels[np.argwhere(self.label == k)] == i) for i in range(10)])
            self.caption[k] = label
        return [self.caption[int(self.label[i])] for i in range(len(self.data))]

if __name__ == '__main__':
    random.seed(0)
    images, labels = load_data()
    images = np.array(images, dtype=np.float)
    labels = np.array(labels, dtype=np.float)
    kmeans = k_means(images, k=10)
    
    while kmeans.max_distance() > 1:
        kmeans.iterate()
        print(kmeans.max_distance())
        print("Accuracy:", np.sum(kmeans.calc_label(labels) == labels) / len(labels))
    
