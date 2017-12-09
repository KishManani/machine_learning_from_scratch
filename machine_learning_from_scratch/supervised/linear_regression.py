import numpy as np
import matplotlib.pyplot as plt


class Regression(object):
    """
    Regression class
    """

    def __init__(self, w=None):
        self.weights = w

    def predict(self, x):
        x = np.insert(x, 0, values=1)  # Add constant 1 to start of array
        assert len(x) == len(self.weights), "Input size of {0} does not match expected size of {1}".format(len(x), len(
            self.weights))
        return np.dot(self.weights.T, x)

    def fit(self, X, y, method='OLS'):
        if method == 'OLS':
            X = np.insert(X, 0, values=1, axis=1)  # Add a column of ones to the start
            ols = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
            self.weights = ols
        elif method == 'SGD':
            X = np.insert(X, 0, values=1, axis=1)  # Add a column of ones to the start
            self.gradient_descent(X, y, epochs=100, learning_rate=0.001)
        return None

    def gradient_descent(self,X,y,epochs,learning_rate):
        num_feats = X.shape[1]
        num_samples = X.shape[0]

        y = y.reshape(num_samples, 1)
        w = np.random.rand(num_feats, 1)

        training_loss_epochs = []
        for ix in range(epochs):
            y_pred = np.dot(X, w)
            err = (y - y_pred)
            dw = np.dot(-X.T, err) / num_samples
            w -= learning_rate * dw
            training_loss = 0.5 * (1 / num_samples) * np.dot(err.T, err)
            if ix % 10 == 0:
                print('epoch {0} : training loss {1}'.format(ix, training_loss))
            training_loss_epochs.append(training_loss[0])
        self.weights = w
        return None


if __name__ == '__main__':
    # Generate random data
    n_feats = 5
    n_obs = 100
    X = np.random.randint(0, 30, size=[n_obs, n_feats])
    y = X[:, 0].reshape(n_obs,1) + np.random.randint(0, 30, size=[n_obs, 1])
    # Fit model
    model = Regression()
    model.fit(X, y, method='SGD')
    # Predict
    y_pred = model.predict(X[0, :])
    print(y_pred)
