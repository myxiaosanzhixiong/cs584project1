import numpy as np
from scipy import linalg

class LassoHomotopyModel():
    """
    Implementation of LASSO regression using the Homotopy method.
    """
    
    def __init__(self, max_iter=1000, tol=1e-6, lambda_max=None, lambda_min_ratio=1e-6):
        """Initialize the LASSO Homotopy model."""
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_max = lambda_max
        self.lambda_min_ratio = lambda_min_ratio

    def fit(self, X, y):
        """Fit the LASSO model using the Homotopy method."""
        # Convert inputs to numpy arrays
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
        
        # Store lambda and coefficient path
        lambda_path = [lambda_current]
        coef_path = [beta.copy()]
        
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
            
            try:
                # Use more robust matrix computation with SVD
                gram_matrix = X_active.T @ X_active
                
                # Add small regularization term for numerical stability
                gram_matrix += np.eye(gram_matrix.shape[0]) * self.tol
                
                # Use pseudoinverse for more stability
                inverse_gram = linalg.pinv(gram_matrix)
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
                        if abs(delta_correlation[j]) > self.tol:  # Threshold to avoid numerical issues
                            gamma1 = (lambda_current - correlation[j]) / (delta_correlation[j])
                            if gamma1 > 0:
                                lambda_gamma.append((gamma1, j, 1))  # 1 means add to active set
                            
                            gamma2 = (lambda_current + correlation[j]) / (delta_correlation[j])
                            if gamma2 > 0:
                                lambda_gamma.append((gamma2, j, -1))  # -1 means add with negative sign
                
                # Calculate step size for variables to leave active set
                beta_gamma = []
                for i, idx in enumerate(active_set):
                    if delta_beta[idx] * active_signs[i] < 0:  # if direction is opposite to sign
                        gamma = -beta[idx] / delta_beta[idx]
                        if gamma > 0:
                            beta_gamma.append((gamma, i, 0))  # 0 means remove from active set
                
                # Combine and sort step sizes
                gamma_list = lambda_gamma + beta_gamma
                if not gamma_list:
                    # No more events, try a small step and check convergence
                    small_step = lambda_current * 0.1
                    beta_new = beta + small_step * delta_beta
                    if np.max(np.abs(beta_new - beta)) < self.tol:
                        break  # Converged
                    else:
                        lambda_current -= small_step
                        beta = beta_new
                else:
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
                
                # Store current lambda and coefficients
                lambda_path.append(lambda_current)
                coef_path.append(beta.copy())
                
                # Check if lambda is small enough to terminate
                if lambda_current <= lambda_min:
                    break
                    
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Matrix computation error: {e}")
                # If computation fails, use current beta and terminate
                break
        
        # Store results
        self.coef_ = beta
        self.active_set_ = active_set
        self.lambda_path_ = np.array(lambda_path)
        self.coef_path_ = np.array(coef_path)
        
        return LassoHomotopyResults(self)

class LassoHomotopyResults():
    """Class to store the results from LASSO Homotopy model fitting."""
    
    def __init__(self, model):
        """Initialize with model parameters."""
        self.coef_ = model.coef_
        self.active_set_ = model.active_set_
        self.lambda_path_ = model.lambda_path_
        if hasattr(model, 'coef_path_'):
            self.coef_path_ = model.coef_path_
    
    def predict(self, X):
        """Predict using the fitted LASSO model."""
        X = np.asarray(X)
        return X @ self.coef_
