# CS584
Illinois Institute of Technology

Name:Jian Zhang
CWID:A20467790

Name:Hisham Mohammed 
Cwid:A20584812

Name: Manthan Surjuse
Cwid: A20588887

# LASSO Regression with Homotopy Method

This repository implements the LASSO (Least Absolute Shrinkage and Selection Operator) regression technique using the Homotopy method. The implementation follows the approach described in ["LASSO Optimization via the Homotopy Method"](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf) by Asif and Romberg.

## What is LASSO Regression?

LASSO is a regression technique that performs both variable selection and regularization. It adds a penalty term (L1 norm of the coefficient vector) to the ordinary least squares objective function:

$$\min_{\beta} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$

Where:
- $X$ is the feature matrix
- $y$ is the target vector
- $\beta$ is the coefficient vector
- $\lambda$ is the regularization parameter

The L1 penalty encourages sparsity in the solution, automatically performing feature selection by driving some coefficients to exactly zero.

## What is the Homotopy Method?

The Homotopy method is an efficient algorithm for computing the entire regularization path for LASSO. It works by:

1. Starting with the highest value of $\lambda$ that makes all coefficients zero
2. Gradually decreasing $\lambda$ and updating the solution along the way
3. Tracking when variables enter or leave the active set (non-zero coefficients)

This approach is computationally efficient because the solution path is piecewise linear, and we can compute exactly when to add or remove variables from the active set.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/myxiaosanzhixiong/lasso-homotopy.git
cd lasso-homotopy
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Basic Usage Example

```python
import numpy as np
from model.LassoHomotopy import LassoHomotopyModel

# Create some example data
X = np.random.randn(100, 10)  # 100 samples, 10 features
true_coef = np.zeros(10)
true_coef[:3] = [1.0, -0.5, 0.25]  # Only first 3 features are non-zero
y = X @ true_coef + 0.1 * np.random.randn(100)  # Add some noise

# Create and fit the model
model = LassoHomotopyModel()
results = model.fit(X, y)

# Get the coefficients
print("Estimated coefficients:", results.coef_)

# Make predictions
X_test = np.random.randn(20, 10)
predictions = results.predict(X_test)
```

### Model Parameters

The `LassoHomotopyModel` class accepts the following parameters:

- `max_iter` (int, default=1000): Maximum number of iterations for the homotopy path
- `tol` (float, default=1e-6): Tolerance for convergence
- `lambda_max` (float, optional): Maximum value of regularization parameter. If None, it will be calculated as the maximum correlation between features and target.
- `lambda_min_ratio` (float, default=1e-3): Ratio of lambda_min/lambda_max where the path ends

### Handling Collinear Data

One of the strengths of LASSO is handling collinear features. For example:

```python
# Create collinear data
X = np.random.randn(100, 20)
X[:, 10:] = X[:, :10] + 0.1 * np.random.randn(100, 10)  # Make last 10 features collinear with first 10

# True coefficients only use first 5 features
true_coef = np.zeros(20)
true_coef[:5] = [1.0, -0.5, 0.25, -0.3, 0.1]
y = X @ true_coef + 0.1 * np.random.randn(100)

# Fit model
model = LassoHomotopyModel()
results = model.fit(X, y)

# Check which coefficients are non-zero
non_zero = np.where(np.abs(results.coef_) > 1e-6)[0]
print("Non-zero coefficient indices:", non_zero)
```

## Running Tests

To run the tests, navigate to the project root directory and run:

```bash
python -m pytest tests/
```

The test suite includes:
- Testing on small datasets
- Testing on collinear data to verify feature selection
- Testing with synthetic data with known coefficients
- Testing with different regularization strengths

## Answers to Project Questions

### What does the model you have implemented do and when should it be used?

The implemented model performs LASSO regression using the Homotopy method. It should be used when:

1. You need to perform both feature selection and regularization simultaneously.
2. You're dealing with high-dimensional data where some features may be irrelevant or redundant.
3. You suspect multicollinearity (strong correlations between predictor variables).
4. You want interpretable models with sparse coefficients (many zeros).

The Homotopy method is particularly useful when you want to efficiently explore the entire regularization path from the most regularized model (all coefficients zero) to less regularized models.

### How did you test your model to determine if it is working reasonably correctly?

I tested the model through several approaches:

1. On synthetic data with known sparse coefficients to see if the model correctly identifies the non-zero coefficients.
2. On collinear data to verify that the model selects a sparse subset of features.
3. With different regularization strengths to confirm that higher regularization leads to more zero coefficients.
4. On small datasets to ensure the model can handle various input shapes and types.

The tests verify that the model correctly performs variable selection, produces reasonably accurate predictions, and has the expected behavior with changes in regularization strength.

### What parameters have you exposed to users of your implementation in order to tune performance?

The implementation exposes several parameters:

1. `max_iter`: Maximum number of iterations for the homotopy path, controlling computational resources.
2. `tol`: Numerical tolerance for convergence.
3. `lambda_max`: Maximum regularization parameter value, where the path begins.
4. `lambda_min_ratio`: Ratio of lambda_min/lambda_max, controlling where the path ends.

Users can tune these parameters to balance model complexity, computational cost, and prediction accuracy.

### Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

The implementation may face challenges with:

1. **Very high-dimensional data**: The current implementation might be slow or memory-intensive with thousands of features. With more time, this could be improved with more efficient matrix operations and sparse matrix support.

2. **Numerical instability**: When features are highly collinear, matrix inversions can become numerically unstable. This could be addressed by implementing more robust matrix inversion techniques like QR decomposition or regularized versions.

3. **Large datasets**: The implementation loads all data into memory, which may not be feasible for very large datasets. A chunking or online learning approach could be implemented to address this.

4. **Convergence issues**: In some cases, the algorithm might require many iterations to converge. This could be addressed by implementing additional stopping criteria or adaptive step sizes.

These limitations are not fundamental to the Homotopy method and could be addressed with additional engineering effort.
