import numpy as np

class LinearRegressionOLS:
    def __init__(self):
        """
        Initializing the Linear Regression model.
        self.intercept:stores the intercept (theta_0)
        self.coefficients: stores the slopes (theta_1,theta_2 .....theta_n)
        """
        self.intercept=None
        self.coefficients=None

    def fit(self, X, y):
        """
        Training the model using the Normal Equation to find the best fit line.

        Normal Equation=    (X_b.T @ X_b)^-1  @ X_b.T @ y 

        Parameters:
        X:Input features with shape (n_samples, n_features)
        y:Target values with shape (n_samples,)
        """
        # Reshaping X if it's a 1D array to make it a 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Adding a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Computing theta using the Normal Equation
        theta = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ y)

        # Extracting the intercept and coefficients
        self.intercept = theta[0]
        self.coefficients = theta[1:]

    def predict(self, X):
        """
        Predict the target values for given input features.

        Parameters:
        X:Input features with shape (n_samples, n_features)

        Returns:
        ndarray:Predicted values based on the trained model.
        """
        # Reshape X if it's a 1D array to make it a 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        #Predicting values using the learned coefficients
        return X @ self.coefficients + self.intercept
