# load thetas from file and predict the output

import numpy as np
import pandas as pd
import sys

def weight_loader(filename):
    normalizing = []
    with open(filename, "r") as f:
        for i in range(6):
            normalizing.append(f.readline())
        thetas = f.readline()
    print("thteas: ", thetas)
    thetas = thetas.split(",")
    thetas = [float(theta) for theta in thetas]

    for i in range(6):
        normalizing[i] = normalizing[i].split(",")
        normalizing[i] = [float(theta) for theta in normalizing[i]]
    return normalizing, thetas

def normalize(X, normalizing):
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - normalizing[i][0]) / (normalizing[i][1] - normalizing[i][0])
    return X


def predict(thetas, X):
    
    new_params = np.zeros(X.shape[0])

    print(X.shape)
    for i in range(X.shape[1]):
        for j in [2,4,8]:
            new_params = np.c_[new_params, np.sin(j * X[:, i]*np.pi)/2 + 1/2, np.cos(j * X[:, i]*np.pi)/2 + 1/2]
    X = np.c_[X, new_params]
    new_params = np.zeros(X.shape[0])   
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            new_params = np.c_[new_params, X[:, i] * X[:, j]]
    X = np.c_[X, new_params]

    # X = X_copy
    X = np.c_[np.ones(X.shape[0]), X]
    print(X.shape)

    # predict
    prediction = np.dot(X, thetas)
    return prediction


if __name__ == "__main__":
    normalizing, thetas = weight_loader("final_weights.csv")
    data = "13	31	5	0	-4	-6"
    arr =  data.split("\t")
    
    X = np.array([arr]).astype(float)
    X = normalize(X, normalizing)
    prediction = predict(thetas, X)
    print(prediction)