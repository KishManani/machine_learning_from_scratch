import numpy as np

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

    def fit(self, X, y, method='OLS', epochs=None, learning_rate=None, batch_size=None):

        if method == 'OLS':
            X = np.insert(X, 0, values=1, axis=1)  # Add a column of ones to the start
            ols = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
            self.weights = ols
        elif method == 'SGD':
            X = np.insert(X, 0, values=1, axis=1)  # Add a column of ones to the start
            self.gradient_descent(X, y, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
        return None

    def gradient_descent(self, X, y, epochs, learning_rate, batch_size=None):

        num_feats = X.shape[1]
        num_samples = X.shape[0]

        y = y.reshape(num_samples, 1)
        w = np.random.rand(num_feats, 1)
        training_loss_epochs = []

        for ix in range(epochs):

            if batch_size: # Mini batch gradient descent
                shuffled_ix = (np.arange(0, len(X)))
                np.random.shuffle(shuffled_ix)
                X=X[shuffled_ix,:]
                y=y[shuffled_ix,:]
                for batch_ix in np.arange(0, X.shape[0], batch_size):
                    w = self.gradient_descent_step(w, X[batch_ix:batch_ix+batch_size], y[batch_ix:batch_ix+batch_size],
                                                   learning_rate)
            else: # Batch gradient descent
                    w = self.gradient_descent_step(w, X, y, learning_rate)

            training_loss = 0.5 * (1 / num_samples) * np.dot((y-np.dot(X,w)).T, (y-np.dot(X,w)))
            if ix % 10 == 0:
                print('epoch {0} : training loss {1}'.format(ix, training_loss))
            training_loss_epochs.append(training_loss[0])

        self.weights = w
        return None

    def gradient_descent_step(self, w, X, y, learning_rate):

        y_pred = np.dot(X, w)
        err = (y - y_pred)
        dw = np.dot(-X.T, err) / len(X)
        w -= learning_rate * dw
        return w


if __name__ == '__main__':
    # Generate random data
    n_feats = 5
    n_obs = 100000
    X = np.random.randint(0, 5, size=[n_obs, n_feats])
    y = X[:, 0].reshape(n_obs, 1) + np.random.randint(0, 30, size=[n_obs, 1])
    # Fit model
    model = Regression()
    model.fit(X, y, method='SGD', batch_size=250, learning_rate=0.0001, epochs=100)
    # Predict
    y_pred = model.predict(X[0, :])
