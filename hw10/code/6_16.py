import numpy as np
import math
import sys
import time
from scipy.spatial import cKDTree


def euclidean(v1, v2):
    return sum((p - q) ** 2 for p, q in zip(v1, v2)) ** .5


def knn(test, xz, y, k):
    true = 0
    false = 0
    distances = []
    for train in xz:
        distances.append(euclidean(test, train))
    points = map(distances.index, heapq.nsmallest(k, distances))
    for i in points:
        if y[i] == 1:
            true += 1
        else:
            false += 1

    return true > false


def uniform_data():
    x1 = []
    x2 = []
    for k in [i for i in range(100)]:
        for m in [j for j in range(100)]:
            x1.append(k / 100)
            x2.append(m / 100)

    return x1, x2


def gaussian_data(n=10000):
    centersX1 = np.random.random_sample((10,))
    centersX2 = np.random.random_sample((10,))
    x1 = []
    x2 = []
    mu, sigma = 0.5, 0.1  # mean and standard deviation
    for i in range(len(centersX1)):
        t1 = np.random.normal(centersX1[i], sigma, (int(n / 10), 1))  # normal distribution for x1
        t2 = np.random.normal(centersX2[i], sigma, (int(n / 10), 1))  # normal distribution for x2
        for i in range(len(t1)):
            x1.append(float(t1[i]))
            x2.append(float(t2[i]))
    return x1, x2


def brute_force(X, test_pts):
    tmp_dist = 100000
    nearest_neighbor = []
    for x in X:
        dist = euclidean(test_pts, x)
        if dist < tmp_dist:
            nearest_neighbor = x
            tmp_dist = dist

    return nearest_neighbor


def branch_n_bound(centers, regions, testPoint):
    tmp_dist = 100000
    ctr_idx = 11
    for i in range(len(centers)):
        dist = euclidean(testPoint, centers[i])
        if dist < tmp_dist:
            ctr_idx = i
            tmp_dist = dist
    nearest_neighbor = brute_force(regions[ctr_idx], testPoint)

    return nearest_neighbor


if __name__ == '__main__':
    option = sys.argv[1]

    if option == 'a':
        train_x1, train_x2 = uniform_data()
    else:
        train_x1, train_x2 = gaussian_data()

    train = [[train_x1[i], train_x2[i]] for i in range(len(train_x1))]
    test = np.random.random_sample((10000, 2))

    #########################################################################################
    startTime = time.time()

    ctr_points = [train[np.random.randint(len(train))]]
    while len(ctr_points) < 10:
        dist_tmp = 0
        list_tmp = []
        for point in train:
            if point in ctr_points:
                continue
            dist = min([euclidean(point, c) for c in ctr_points])
            if dist > dist_tmp:
                list_tmp = point
                dist_tmp = dist
        ctr_points.append(list_tmp)

    regions = [[] for i in range(len(ctr_points))]
    tree = cKDTree(ctr_points)
    for train_point in train:
        _, region_idx = tree.query(train_point)
        regions[region_idx].append(train_point)

    for test_pts in test:
        branch_n_bound(ctr_points, regions, test_pts)

    endTime = time.time()
    print('Running time for NN with branch and bound: {}'.format(endTime - startTime))

    #########################################################################################
    startTime = time.time()

    for test_pts in test:
        brute_force(train, test_pts)

    endTime = time.time()
    print('Running time for NN with brute force: {}'.format(endTime - startTime))

    #########################################################################################
