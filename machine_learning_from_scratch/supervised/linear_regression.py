import numpy as np


class LinearRegression(object):

    def __init__(self, W=None):
        self.weights = W
        self.training_loss = None  # Stores training loss after running gradient descent

    def predict(self, X):

        X = np.insert(X, 0, values=1, axis=1)  # Add constant 1 to start of array

        assert X.shape[1] == len(self.weights), "Input size of {0} does not match expected size of {1}".format(
            X.shape[1], len(self.weights))
        return np.dot(X, self.weights)

    def fit(self, X, y, method='OLS', epochs=None, learning_rate=None, batch_size=None):

        if method == 'OLS':
            X = np.insert(X, 0, values=1, axis=1)  # Add a column of ones to the start
            ols = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
            self.weights = ols
        elif method == 'SGD':
            X = np.insert(X, 0, values=1, axis=1)  # Add a column of ones to the start
            self.gradient_descent(X, y, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
        return None

    def gradient_descent(self, X, y, epochs, learning_rate, batch_size):

        num_feats = X.shape[1]
        num_samples = X.shape[0]

        y = y.reshape(num_samples, 1)
        W = np.random.rand(num_feats, 1)
        training_loss_epochs = []

        for ix in range(epochs):
            shuffled_ix = (np.arange(0, len(X)))
            np.random.shuffle(shuffled_ix)
            X = X[shuffled_ix, :]
            y = y[shuffled_ix, :]

            for batch_ix in np.arange(0, X.shape[0], batch_size):
                W = self.gradient_descent_step(W, X[batch_ix:batch_ix + batch_size],
                                               y[batch_ix:batch_ix + batch_size],
                                               learning_rate)

            training_loss = (0.5 / num_samples) * np.dot((y - np.dot(X, W)).T, (y - np.dot(X, W)))

            if ix % 10 == 0:
                print('epoch {0} : training loss {1}'.format(ix, training_loss))
                training_loss_epochs.append(training_loss[0])

        self.weights = W
        self.training_loss = training_loss_epochs
        return None

    @staticmethod
    def gradient_descent_step(W, X, y, learning_rate):

        y_pred = np.dot(X, W)
        err = (y - y_pred)
        dW = np.dot(-X.T, err) / len(X)
        W -= learning_rate * dW
        return W


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Generate random data
    n_feats = 1
    n_obs = 10
    W = np.random.random((n_feats, 1))
    X = np.random.randint(0, 100, size=[n_obs, n_feats])
    y = np.dot(X, W) + np.random.rand(n_obs, 1) * 10

    # Fit model
    model = Regression()
    model.fit(X, y, method='SGD', batch_size=250, learning_rate=0.0001, epochs=100)

    # Predict
    y_pred = model.predict(X)

    # Plot results
    ax, fig = plt.subplots(figsize=[10, 7.5])
    plt.scatter(X, y)
    plt.plot(X, y_pred)
    plt.show()
