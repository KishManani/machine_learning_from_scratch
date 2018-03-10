import numpy as np


class LogisticRegression(object):
    """
    Logistic regression model.
    """

    def __init__(self, W=None):
        self.weights = W
        self.training_loss = None  # Stores training loss after running gradient descent

    def fit(self, X, y, method='SGD', epochs=None, learning_rate=None, batch_size=None):
        """
        Fit linear regression model given training data and target array

        Parameters
        ----------
        X : np.array
            Input data where rows are observations and columns are features
        y : np.array
            Target values
        method : str
            'SGD' for batch gradient descent
        epochs : int
            Number of epochs to run gradient descent
        learning_rate : float
            Learning rate for gradient descent
        batch_size : int
            Batch size for batch gradient descent

        Returns
        -------
        None

        """
        X = np.insert(X, 0, values=1, axis=1)

        if method == 'SGD':
            self._gradient_descent(X, y, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

        return None

    def predict(self, X):
        X = np.insert(X, 0, values=1, axis=1)
        y_pred = self.sigmoid(np.dot(X, self.weights))
        return y_pred

    def _gradient_descent(self, X, y, epochs, learning_rate, batch_size):
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
                W = self._gradient_descent_step(W,
                                                X[batch_ix:batch_ix + batch_size],
                                                y[batch_ix:batch_ix + batch_size],
                                                learning_rate)

            y_pred = self.sigmoid(np.dot(X, W))
            training_loss = self.logloss(y, y_pred)

            if ix % 10 == 0:
                print('epoch {0} : training loss {1}'.format(ix, training_loss))
                training_loss_epochs.append(training_loss[0])

        self.weights = W
        self.training_loss = training_loss_epochs
        return None


        return None

    def _gradient_descent_step(self):
        return None

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(x))

    @staticmethod
    def logloss(y, y_pred):
        loss = 1
        return loss

if __name__ == '__main__':
    from sklearn import datasets
    X, y = datasets.make_classification(n_samples=1000, n_features=5,
                                        n_informative=2, n_redundant=3,
                                        random_state=42)
    # Fit model
    model = LogisticRegression()
    model.fit(X, y, method='SGD', batch_size=250, learning_rate=0.0001, epochs=100)

    # Predict
    y_pred = model.predict(X)