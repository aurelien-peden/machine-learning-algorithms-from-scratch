import numpy as np


class LinearRegression(object):
    def __init__(self, learning_rate=0.001, epochs=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.parameters = None

    def fit(self, X, y):
        m, n_x = X.shape

        self.parameters = self._initialize_parameters(n_x)

        for _ in range(self.epochs):
            y_hat = np.dot(X, self.parameters['W']) + self.parameters['b']

            grads = self._compute_gradients(m, y_hat, X, y)

            self._update_parameters(grads)

    def _initialize_parameters(self, n_x):
        parameters = {}

        parameters['W'] = np.zeros(n_x)
        parameters['b'] = 0

        return parameters

    def _compute_gradients(self, m, y_hat, X, y):
        grads = {}

        grads['dW'] = (1 / m) * np.dot(X.T, (y_hat - y))
        grads['db'] = (1 / m) * np.sum(y_hat - y)

        return grads

    def _update_parameters(self, grads):
        self.parameters['W'] -= self.learning_rate * grads['dW']
        self.parameters['b'] -= self.learning_rate * grads['db']

    def predict(self, X):
        y_hat = np.dot(X, self.parameters['W']) + self.parameters['b']
        return y_hat

    def compute_cost(self, y, y_hat):
        return np.mean((y - y_hat)**2)
