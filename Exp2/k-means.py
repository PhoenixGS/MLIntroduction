import torch
from sklearn.manifold import TSNE
import numpy as np
import random
from torchvision import datasets
import matplotlib.pyplot as plt

def load_data(): # load MNIST data
    train_data = datasets.MNIST(root = './data', train=True, download=True)
    images = train_data.data.numpy()
    labels = train_data.targets.numpy()
    return images, labels

def pr(image): # print image
    for i in range(len(image)):
        for j in range(len(image[i])):
            print('.' if image[i][j] < 128 else '#', end='')
        print()

class k_means:
    def __init__(self, data, k=10): # initialize k-means
        self.data = data
        self.k = k
        index = np.array(random.sample(range(len(data)), k))
        self.centers = data[index]
        self.label = np.zeros(len(data))
        self.update_label()

    def update_label(self):
        count = 0
        for i in range(len(self.data)): # update labels of each sample
            distances = [np.linalg.norm(self.data[i] - self.centers[c], ord=np.inf) for c in range(self.k)]
            cluster = np.argmin(distances)
            count += (self.label[i] != cluster)
            self.label[i] = cluster
        return count <= len(self.data) // 1000 # return True if the number of samples whose labels are changed is less than 0.1% of the total number of samples
    
    def iterate(self):
        for c in range(self.k): # update centers of clusters
            index = np.argwhere(self.label == c)
            index = index.reshape(index.shape[0])
            center = np.mean(self.data[index], axis=0)
            self.centers[c] = center
        return self.update_label() # update labels

    def max_distance(self):
        maxdistance = 0
        for i in range(len(self.data)): # calculate the max distance between each sample and its center
            distance = np.linalg.norm(self.data[i] - self.centers[int(self.label[i])])
            if distance > maxdistance:
                maxdistance = distance
        return maxdistance

    def calc_label(self, labels):
        self.caption = np.zeros(self.k)
        for k in range(self.k): # calculate the label of each cluster
            label = np.argmax([np.sum(labels[np.argwhere(self.label == k)] == i) for i in range(10)])
            self.caption[k] = label
        return [self.caption[int(self.label[i])] for i in range(len(self.data))] # return the label of each sample

if __name__ == '__main__':
    random.seed(1)
    images, labels = load_data()
    images = np.array(images, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    kmeans = k_means(images, k=10) # initialize k-means
    
    while True:
        if kmeans.iterate():
            break
        print(kmeans.max_distance()) # output max distance each iteration
        print("Accuracy:", np.sum(kmeans.calc_label(labels) == labels) / len(labels)) # output accuracy each iteration


    x_low = TSNE(n_components=3, random_state=0).fit_transform(images[:10000].reshape(len(images[:10000]), -1)) # t-SNE

    # output wrong cases
    # y = kmeans.calc_label(labels)
    # wrong_idx = np.argwhere(y != labels)
    # for i in wrong_idx[:10]:
    #     print(y[i[0]], labels[i[0]])
    #     pr(images[i[0]])
    #     print()

    # unique y
    print(np.unique(y, return_inverse=True)[0])

    # 2-d plot
    # plt.scatter(x_low[:, 0], x_low[:, 1], c=y, cmap='tab10')
    # plt.show()

    # 3-d plot
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_low[:,0], x_low[:,1], x_low[:,2] , c=y[:10000], s=10) # select 10000 samples to plot
    plt.show()
