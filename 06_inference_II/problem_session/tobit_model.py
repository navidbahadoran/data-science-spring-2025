import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

class TobitModel:
    def __init__(self, X, y, ul=np.inf): 
        self.X = X    # Independent variables
        self.y = y    # Dependent variable
        self.ul = ul  # Upper limit for censoring
        self.params_ = None # Placeholder for estimated parameters

    def tobit_ll(self, par):
        '''Computes negative log likelihood as a function of the parameters'''
        X = self.X
        y = self.y
        ul = self.ul
        
        # Extract parameters
        sigma = np.exp(par[-1])
        beta = par[:-1]
        
        indicator = (y < ul)
        # Linear predictor
        lp = np.dot(X, beta)
        
        # A small epsilon to prevent log(0)
        eps = 1e-10
        
        # Compute likelihood parts, clipping to avoid log(0)
        ll_obs = np.log(np.clip((1/sigma) * norm.pdf((y - lp) / sigma), eps, None))
        ll_cens = np.log(np.clip(1 - norm.cdf((ul - lp) / sigma), eps, None))
        
        ll = np.sum(indicator * ll_obs) + np.sum((1 - indicator) * ll_cens)
        return -ll
        
    def fit(self):
        """Fit the Tobit model using maximum likelihood estimation."""
        num_params = self.X.shape[1] + 1  # Number of parameters (beta + sigma)
        
        # Initialize the parameters with OLS linear regression model parameters
        lr = LinearRegression(fit_intercept=False)
        lr.fit(self.X, self.y)
        params = list(lr.coef_)
        
        # Compute sigma from residuals and ensure it is not too small
        residual_std = np.sqrt(np.sum((self.y - lr.predict(self.X))**2) / len(self.y))
        sigma = residual_std * 1.5
        sigma = max(sigma, 1e-5)  # Enforce a lower bound for sigma
        params.append(np.log(sigma))
        self.params_ = params

        # Correct bounds: for the log(sigma) parameter, lower bound should be log(1e-5)
        bounds = [(None, None)] * len(params[:-1]) + [(np.log(1e-5), None)]
        
        # Minimize the negative log likelihood
        result = minimize(self.tobit_ll, self.params_, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 10000, 'ftol': 1e-10})
        self.params_ = result.x  # Store the optimized parameters

    def predict(self, X_new):
        """
        Predict the latent variable values for new data.
        """
        if self.params_ is None:
            raise ValueError("Model is not fitted yet. Please call the fit method first.")

        beta = self.params_[:-1]         # Coefficients
        sigma = np.exp(self.params_[-1])   # Standard deviation
        
        # Linear predictor
        lp = np.dot(X_new, beta)
        return lp  # Return the predicted latent variable values
