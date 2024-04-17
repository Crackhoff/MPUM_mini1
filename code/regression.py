import numpy as np
import pandas as pd
import params
from MPUM_mini1.code.helper import *

class LinearRegression:
    def __init__(self):
        self.theta = None
        self.Y = None
        self.X = None
        self.etha = params.ETHA
        self.epsilon = params.EPSILON
        self.iterations = params.ITERATIONS
    
    def _cost(self, X, Y):
        return compute_loss(Y, self.predict(X))
        
    def _set(self, X, Y, is_random = False):
        self.X = X
        self.Y = Y
        if is_random:
            self.theta = np.random.rand(X.shape[1])
        else:
            self.theta = np.zeros(X.shape[1])
        
    def predict(self, X):
        return np.dot(X, self.theta)
    
    def _step_gradient(self, X, Y, theta, etha):
        m = X.shape[0]
        prediction = np.dot(X, theta)
        gradient =  (1/m) * etha * np.dot(X.T, prediction - Y)
        theta = theta - etha * gradient
        gradient_len = np.linalg.norm(gradient)
        return theta, gradient_len < self.epsilon
    
    def gradient_descent(self):
        history = []
        for i in range(self.iterations):
            self.theta, converged = self._step_gradient(self.X, self.Y, self.theta, self.etha)
            history.append(self.theta)
            # if converged:
            #     print("converged")
            #     break
        return self.theta, history
    
    def fit(self,X, Y):
        self._set(X, Y)
        return self.gradient_descent()
    
    def weights(self):
        return self.theta



class MiniBatchLinearRegression(LinearRegression):
    def __init__(self, batch_size=10):
        super().__init__()
        self.batch_size = batch_size
        
    def _get_batches(self, X, Y, batch_size):
        m = len(Y)
        indices = np.random.permutation(m)
        X = X[indices]
        Y = Y[indices]
        batches = []
        for start_idx in range(0, X.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, X.shape[0])
            batch = slice(start_idx, end_idx)
            batches.append((X[batch], Y[batch]))
        return batches
    
    def gradient_descent(self):
        m = len(self.Y)
        history = []
        batches = self._get_batches(self.X, self.Y, self.batch_size)
        
        for i in range(self.iterations):
            cost = 0.0
            conv = 0
            for X_batch, Y_batch in batches:
                self.theta, converged = self._step_gradient(X_batch, Y_batch, self.theta, self.etha)
                cost += self._cost(X_batch, Y_batch)
                # convergence is nice, but harder to compare
                # if converged:
                    # conv += 1
            history.append(self.theta)
            # if conv == self.batch_size:
            #     print("converged")
            #     break
            
        return self.theta, history
    
        
    def _step_gradient(self, X, Y, theta, etha):
        m = X.shape[0]
        batch_size = int(m * self.batch_size)
        idx = np.random.randint(m, size=batch_size)
        gradient = np.dot(X[idx].T, (np.dot(X[idx], theta) - Y[idx])) / batch_size
        theta = theta - etha * gradient
        gradient_len = np.linalg.norm(gradient)
        return theta, gradient_len < self.epsilon

def StochasticLinearRegression():
    return MiniBatchLinearRegression(1)

class RidgeRegression(LinearRegression):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        
    def _step_gradient(self, X, Y, theta, etha):
        m = X.shape[0]
        regularizer = np.concatenate(([0], theta[1:]))
        gradient = np.dot(X.T, (np.dot(X, theta) - Y)) / m + self.alpha * regularizer
        theta = theta - etha * gradient
        gradient_len = np.linalg.norm(gradient)
        return theta, gradient_len < self.epsilon
    
class LassoRegression(LinearRegression):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        
    def _step_gradient(self, X, Y, theta, etha):
        m = X.shape[0]
        regularizer = np.concatenate(([0], np.sign(theta[1:])))
        gradient = np.dot(X.T, (np.dot(X, theta) - Y)) / m + self.alpha * regularizer
        theta = theta - etha * gradient
        gradient_len = np.linalg.norm(gradient)
        return theta, gradient_len < self.epsilon
    
class ElasticNetRegression(LinearRegression):
    def __init__(self, alpha, l1_ratio):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
    def _step_gradient(self, X, Y, theta, etha):
        m = X.shape[0]
        gradient = np.dot(X.T, (np.dot(X, theta) - Y)) / m
        base_change = gradient
        l1 = np.concatenate(([0], np.sign(theta[1:])))
        l2 = np.concatenate(([0], theta[1:])) * (1 - self.l1_ratio)
        change = base_change + self.alpha * (self.l1_ratio *l1 + (1-self.l1_ratio) *l2)
        theta = theta - etha * change
        change_len = np.linalg.norm(change)
        return theta, change_len < self.epsilon