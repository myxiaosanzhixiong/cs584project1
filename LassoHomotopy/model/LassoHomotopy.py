import numpy as np

class LassoHomotopyModel():
    """
    Implementation of LASSO regression using the Homotopy method.
    
    The Homotopy method computes the entire regularization path for LASSO,
    efficiently finding solutions for all values of the regularization parameter lambda.
    
    References:
    - "Homotopy Path Following for the LASSO" by Osborne et al.
    - "LASSO Optimization via the Homotopy Method" by Asif and Romberg
    """
    
    def __init__(self, max_iter=1000, tol=1e-6, lambda_max=None, lambda_min_ratio=1e-3):
        """
        Initialize the LASSO Homotopy model.
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations for the homotopy path
        tol : float
            Tolerance for convergence
        lambda_max : float, optional
            Maximum value of regularization parameter
        lambda_min_ratio : float
            Ratio of lambda_min/lambda_max where the path ends
        """
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_max = lambda_max
        self.lambda_min_ratio = lambda_min_ratio

    def fit(self, X, y):
        """
        Fit the LASSO model using the Homotopy method.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples, 1) or (n_samples,)
            Target values
            
        Returns:
        --------
        results : LassoHomotopyResults
            Object containing the results and trained model
        """
        # Convert inputs to numpy arrays and handle different input shapes
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Reshape y if needed
        if y.ndim > 1:
            y = y.ravel()
        
        n_samples, n_features = X.shape
        
        # Compute lambda_max if not provided
        correlation = np.abs(X.T @ y)
        if self.lambda_max is None:
            self.lambda_max = np.max(correlation)
        
        # Initialize lambda and active set
        lambda_current = self.lambda_max
        lambda_min = self.lambda_max * self.lambda_min_ratio
        
        # Initialize beta (coefficients) to zeros
        beta = np.zeros(n_features)
        
        # Initialize active set and its sign
        active_set = []
        active_signs = []
        
        # Homotopy algorithm
        for iteration in range(self.max_iter):
            # Compute residual
            residual = y - X @ beta
            
            # Calculate correlations
            correlation = X.T @ residual

            # If first iteration or if we need to add variables to active set
            if len(active_set) == 0:
                # Find feature with maximum absolute correlation
                j = np.argmax(np.abs(correlation))
                active_set.append(j)
                active_signs.append(np.sign(correlation[j]))
            
            # Active set matrix
            X_active = X[:, active_set]
            
            # Calculate direction using active set
            signs = np.array(active_signs)
            X_active_signs = X_active * signs.reshape(1, -1)
            
            # Compute the direction using matrix operations
            try:
                # Try using the normal equations
                gram_matrix = X_active.T @ X_active
                inverse_gram = np.linalg.inv(gram_matrix)
                direction = inverse_gram @ signs
                
                # Compute the direction in the feature space
                delta_beta = np.zeros(n_features)
                for i, idx in enumerate(active_set):
                    delta_beta[idx] = direction[i]
                
                # Compute step sizes for lambda decrease
                delta_correlation = X.T @ (X @ delta_beta)
                
                # Calculate step size for variables to enter
                lambda_gamma = []
                for j in range(n_features):
                    if j not in active_set:
                        if abs(delta_correlation[j]) > 0:
                            gamma = (lambda_current - correlation[j]) / (delta_correlation[j])
                            if gamma > 0:
                                lambda_gamma.append((gamma, j, 1))  # 1 means add to active set
                            
                            gamma = (lambda_current + correlation[j]) / (delta_correlation[j])
                            if gamma > 0:
                                lambda_gamma.append((gamma, j, -1))  # -1 means add with negative sign
                
                # Calculate step size for variables to leave active set
                beta_gamma = []
                for i, idx in enumerate(active_set):
                    if delta_beta[idx] * active_signs[i] < 0:  # if direction is opposite to sign
                        gamma = -beta[idx] / delta_beta[idx]
                        if gamma > 0:
                            beta_gamma.append((gamma, i, 0))  # 0 means remove from active set
                
                # Combine and sort step sizes
                if not lambda_gamma and not beta_gamma:
                    break  # No more events, exit
                
                gamma_list = lambda_gamma + beta_gamma
                if not gamma_list:
                    break
                
                min_gamma, min_idx, min_type = min(gamma_list)
                
                # Update beta
                beta += min_gamma * delta_beta
                
                # Update lambda
                lambda_current -= min_gamma
                
                # Update active set
                if min_type == 0:  # Remove from active set
                    i_remove = min_idx
                    idx_remove = active_set[i_remove]
                    beta[idx_remove] = 0  # Zero out coefficient
                    
                    # Remove from active set and signs
                    active_set.pop(i_remove)
                    active_signs.pop(i_remove)
                    
                elif min_type in [1, -1]:  # Add to active set
                    j_add = min_idx
                    active_set.append(j_add)
                    active_signs.append(min_type)
                
                # Check if lambda is small enough to terminate
                if lambda_current <= lambda_min:
                    break
                    
            except np.linalg.LinAlgError:
                # If matrix inversion fails, handle the error
                break
        
        # Store results
        self.coef_ = beta
        self.active_set_ = active_set
        self.lambda_path_ = np.array([lambda_current])
        
        return LassoHomotopyResults(self)

class LassoHomotopyResults():
    """
    Class to store the results from LASSO Homotopy model fitting.
    Provides methods for prediction and accessing model parameters.
    """
    
    def __init__(self, model):
        """
        Initialize with model parameters.
        
        Parameters:
        -----------
        model : LassoHomotopyModel
            The fitted LASSO Homotopy model
        """
        self.coef_ = model.coef_
        self.active_set_ = model.active_set_
        self.lambda_path_ = model.lambda_path_
    
    def predict(self, X):
        """
        Predict using the fitted LASSO model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Returns predicted values.
        """
        X = np.asarray(X)
        return X @ self.coef_
