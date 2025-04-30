import numpy as np

class RidgeRegression:

    def __init__(self,alpha=0.1):
        self.alpha=alpha
        self.coefficients=None
        self.intercept=None

    def fit(self,X,y):

        # Reshape if input is 1D to make it 2D
        if X.ndim == 1:
            X=X.reshape(-1, 1)

        m,n=X.shape

        # Adding a column of 1s to X for intercept term
        X_b=np.c_[np.ones((m, 1)), X]

        # Creating identity matrix of size (n+1) for regularization
        I=np.eye(n+1)
        I[0,0]=0  # Intercept will not be regularized

        # Ridge Regression θ = (XᵀX + lambda)^(-1) Xᵀy
        self.theta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y

        # Extracting intercept and coefficients
        self.intercept=self.theta[0]
        self.coefficients=self.theta[1:]

    def predict(self, X):
        # Reshape input if it's a 1D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Predicting  y = X * coef + intercept
        return X @ self.coefficients + self.intercept