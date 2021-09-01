import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # x_train -> its a numpy vector of size m*n where m:#of samples, n:#of features
    def fit(self, x_train, y_train):
        samples, features = x_train.shape
        self.weights = np.zeros(features)
        self.bias = 0
        # Perform Gradient Descent
        for _ in range(self.n_iters):
            # calculate linear model value first
            linear_model = np.dot(self.weights, x_train) + self.bias
            # apply sigmod function
            y_predicted = self.sigmoid(linear_model)

    def predict(self, x_test):
        pass

    def sigmoid(self, linear_model_result):
        return 1 / (1 + np.exp(linear_model_result))
