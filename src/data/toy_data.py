import numpy as np
import random 

from sklearn import datasets
from sklearn.model_selection import train_test_split


def bubbles(mu1, mu2, sigma, n, target_class, seed):

    np.random.seed(seed)
    random.seed(seed)
    y = target_class*np.ones(int(n))
    X1 = np.random.normal(mu1, sigma, size=[int(n), 1])
    X2 = np.random.normal(mu2, sigma, size=[int(n), 1])

    X = np.hstack([X1, X2])

    return X, y.astype(int)

def two_normals(n=1000 , prop_dense=0.1, prop_test= 0.1, seed=1234):
    
    X_0, y_0 = bubbles(0, 0, 2, n*(1-prop_dense), 0, seed)
    X_1, y_1 = bubbles(0.5, 1, 0.25, n*prop_dense, 1, seed)
    X = np.vstack((X_0, X_1))
    y = np.hstack((y_0, y_1))
    
    X_test_0, y_test_0 = bubbles(0, 0, 5, n*(1-prop_dense)*prop_test, 0, seed+seed)
    X_test_1, y_test_1 = bubbles(0.5, 1, 0.25, n*prop_dense*prop_test, 1, seed+seed)
    X_test = np.vstack((X_test_0, X_test_1))
    y_test = np.hstack((y_test_0, y_test_1))
    return X, X_test, y, y_test

def half_moons(n=1000, prop_test=0.1, noise=0.1, seed=1234, train_test=False):
    
    np.random.seed(seed)
    random.seed(seed)
    
    X, y = datasets.make_moons(n_samples=n, noise=noise, random_state=seed)
    
    if train_test:
        sets = train_test_split(X, y, random_state = seed, test_size=prop_test, stratify=y)
    else:
        sets = X, y
    
    return sets

def normal(n=1000, prop_test=0.1, noise=0.5, centers=12, seed=1234, train_test=False):
    
    
    np.random.seed(seed)
    random.seed(seed)
    
    X, y = datasets.make_blobs(n_samples=n, centers=centers, n_features=2, random_state=seed, cluster_std=noise)
    y = y % 2
    
    if train_test:
        sets = train_test_split(X, y, random_state = seed, test_size=prop_test, stratify=y)
    else:
        sets = X, y
    
    return sets


