import matplotlib.pyplot as plt
import numpy as np
import params
    
# plotting each feature to value
def plot_features_to_value(df):
    '''
    last column is the target
    plot the feature to the value
    grid for 6 features
    '''
    fix, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax = ax.ravel()
    for i in range(6):
        ax[i].scatter(df[i], df[df.columns[-1]])
        ax[i].set_xlabel("feature " + str(i))
        ax[i].set_ylabel("value")

def compute_loss(y, y_hat):
    # y - true value
    # y_hat - predicted value
    return compute_MSE(y, y_hat)
    # return compute_MAE(y, y_hat)
    # return compute_Huber_loss(y, y_hat, params.DELTA)

def compute_MAE(y, y_hat):
    return abs(y - y_hat).mean()
 
def compute_MSE(y, y_hat):
    return ((y - y_hat) ** 2).mean()

def compute_Huber_loss(y, y_hat, delta):
    return delta**2 * (np.sqrt(1 + ((y - y_hat) / delta) ** 2) - 1).mean()

# unused, but that is also a way to normalize the data
standardize = lambda x: (x - x.mean()) / x.std() 

normalize_lambda = lambda x: (x - x.min()) / (x.max() - x.min())

# standardize and normalize for the data, x is a DataFrame object
def normalize(data):
    '''
    function takes data as an input and returns normalized data
    '''
    return data.apply(normalize_lambda)



def split_data(X, Y, ratio=0.8):
    '''
    function takes data and ratio
    and returns the split data
    X is the input data
    Y is the output data
    ratio is the split ratio
    X_train is the data used for training
    Y_train is the corresponding output for X_train
    X_test is the rest of the data
    Y_test is the corresponding output for X_test
    '''
    # shuffle the data
    idx = np.random.permutation(len(X))
    X = X[idx]
    Y = Y[idx]
    # split the data
    n = int(len(X) * ratio)
    X_train = X[:n]
    X_test = X[n:]
    Y_train = Y[:n]
    Y_test = Y[n:]
    return X_train, X_test, Y_train, Y_test


def plot_cost_history(cost_histories):
    for cost_history in cost_histories:
        plt.plot(cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost History")
    plt.show()