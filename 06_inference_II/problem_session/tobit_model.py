import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

class TobitModel:
    def __init__(self, X,y, ul=np.inf): 
        self.X = X    # Independent variables
        self.y = y  # Dependent variable
        self.ul = ul        # Upper limit for censoring
        self.params_ = None # Placeholder for estimated parameters

    def tobit_ll(self, par):
        '''Computes negative log likelihood as a function of the parameters'''
        X = self.X
        y = self.y
        ul = self.ul
        # parameters
        sigma = np.exp(par[-1]) 
        beta = par[:-1]
        
        indicator = (y < ul)  
        # linear predictor
        lp = np.dot(X, beta)
        
        # log likelihood
        ll = np.sum(indicator * np.log((1/sigma) * norm.pdf((y - lp) / sigma))) + np.sum((1 - indicator) * np.log(1 - norm.cdf((ul - lp) / sigma)))

        return -ll
        
    def fit(self):
        """Fit the Tobit model using maximum likelihood estimation."""
        num_params = self.X.shape[1] + 1  # Number of parameters (beta + sigma)

        # Initialize the parameters with OLS linear regression model parameters
        lr = LinearRegression(fit_intercept=False)
        lr.fit(self.X, self.y)
        params = list(lr.coef_)
        sigma = np.sqrt(np.sum((self.y - lr.predict(self.X))**2) / len(self.y)) * 1.5  # Adjust sigma initialization
        params.append(np.log(sigma))
        self.params_ = params

        # Bound for sigma to ensure positivity
        bounds = [(None, None)] * len(params[:-1]) + [(1e-5, None)]  # Small lower bound for sigma

        # minimize the loss function
        result = minimize(self.tobit_ll, self.params_, method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000, 'ftol': 1e-10})
        self.params_ = result.x  # Store the optimized parameters


    def predict(self, X_new):
        """
        Predict the latent variable values for new data.

        Parameters:
        X_new: np.ndarray
            New independent variable data to make predictions.

        Returns:
        np.ndarray
            Predicted values for the latent variable.
        """
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Please call the fit method first.")

        beta = self.params_[:-1]  # Coefficients
        sigma = np.exp(self.params_[-1])  # Standard deviation

        # Linear predictor
        lp = np.dot(X_new, beta)

        return lp  # Return the predicted latent variable values