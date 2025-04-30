import numpy as np

class LinearRegressionGD:
    def __init__(self,epoch=1000,learning_rate=0.1):
        # Initializing the number of times model should learn the data (epochs) and learning rate for gradient descent
        self.epoch=epoch
        self.learning_rate=learning_rate
        self.coefficients=None 
        self.intercept=None  

    def fit(self,X,y):
        m=len(X)  #Number of training samples

        # If X is a 1D array, reshape it to a 2D array (n_samples, 1)
        if X.ndim == 1:
            X=X.reshape(-1, 1)

        # Add a column of ones to X for the intercept
        X_b=np.c_[np.ones(X.shape[0]), X]

        # Initialize zero for intercept and coefficients
        self.theta=np.zeros(X_b.shape[1])

        # Gradient descent for the number of epochs
        for _ in range(self.epoch):
            # Gradient (derivative of the cost function)
            gradient=(2 / m) * X_b.T @ (X_b @ self.theta - y)
            
            # Updating theta based on the gradient and learning rate
            self.theta-=self.learning_rate*gradient

        # After fitting, extract intercept and coefficients from theta
        self.intercept=self.theta[0]  
        self.coefficients=self.theta[1:] 

    def predict(self,X):
        # If X is a 1D array, reshape it to a 2D array (n_samples, 1)
        if X.ndim==1:
            X=X.reshape(-1, 1)

        # Predicting the values using the learned coefficients and intercept
        return X @self.coefficients+self.intercept
