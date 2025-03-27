# CS584
Illinois Institute of Technology

Name:Jian Zhang
CWID:A20467790

Name:Hisham Mohammed 
Cwid:A20584812

Name: Manthan Surjuse
Cwid: A20588887

#Test Result 
<img width="1362" alt="Screenshot 2025-03-12 at 4 07 07 PM" src="https://github.com/user-attachments/assets/6983b4d3-04d4-4744-a864-35a0480d532d" />

# LASSO Regression Using Homotopy Method

This project implements LASSO (Least Absolute Shrinkage and Selection Operator) regression using the Homotopy method. The implementation follows the approach described in ["LASSO Optimization via the Homotopy Method"](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf) by Asif and Romberg.

## Introduction to LASSO and Homotopy Method

LASSO regression is a popular linear regression technique with L1 regularization. Its optimization objective is:

$$\min_{\beta} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$

where the L1 penalty encourages sparsity by driving some coefficients to exactly zero, thus performing feature selection.

The Homotopy method is an efficient algorithm for computing the entire regularization path for LASSO. It works by:
1. Starting from the highest λ value that makes all coefficients zero
2. Gradually decreasing λ and updating the solution along the way
3. Tracking when variables enter or leave the active set (non-zero coefficients)

Since the solution path is piecewise linear, this approach allows for exact computation of when to add or remove variables from the active set, making it computationally efficient.

## Installation and Setup

```bash
# Clone the repository
git clone [https://github.com/myxiaosanzhixiong/lasso-homotopy.git](https://github.com/myxiaosanzhixiong/cs584project1/edit/main/README.md)
cd lasso-homotopy

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

```python
import numpy as np
from model.LassoHomotopy import LassoHomotopyModel

# Create sample data
X = np.random.randn(100, 10)  # 100 samples, 10 features
y = X[:, :3] @ np.array([3.0, -1.5, 2.0]) + 0.5 * np.random.randn(100)  # Only first 3 features are relevant

# Fit the model
model = LassoHomotopyModel(standardize=True)
results = model.fit(X, y)

# View results
print("Coefficients:", results.coef_)
print("Active features:", results.active_set_)
print("R^2 score:", results.score(X, y))

# Make predictions
new_data = np.random.randn(5, 10)
predictions = results.predict(new_data)
```

### Handling Collinear Data

```python
# Create collinear data
X = np.random.randn(100, 20)
X[:, 10:] = X[:, :10] + 0.1 * np.random.randn(100, 10)  # Last 10 features highly correlated with first 10
y = X[:, :5] @ np.array([2.0, -1.0, 1.5, -0.5, 1.0]) + 0.2 * np.random.randn(100)

# Fit model with standardization
model = LassoHomotopyModel(standardize=True, lambda_min_ratio=1e-5)
results = model.fit(X, y)

# Check which coefficients were selected
non_zeros = np.where(np.abs(results.coef_) > 1e-6)[0]
print("Selected feature indices:", non_zeros)
```

## Model Parameters

The `LassoHomotopyModel` class accepts the following parameters:

- `max_iter` (default=1000): Maximum number of iterations for the homotopy path
- `tol` (default=1e-6): Tolerance for convergence
- `lambda_max` (optional): Maximum value of regularization parameter. If None, calculated as the maximum correlation between features and target
- `lambda_min_ratio` (default=1e-6): Ratio of lambda_min/lambda_max where the path ends
- `standardize` (default=False): Whether to standardize features before fitting

## Running Tests

```bash
# From project root directory
python -m pytest tests/

# Or from tests directory
cd tests
pytest
```

The test suite includes tests on small datasets, collinear data, different regularization strengths, and more.

## Project Questions

### What does the model you have implemented do and when should it be used?

I've implemented a LASSO regression model using the Homotopy method, which is a linear regression technique that performs both feature selection and coefficient estimation by introducing an L1 regularization term.

This model is particularly useful in the following scenarios:
- When you need to automatically select important features from a large set
- When dealing with high-dimensional data (especially when features approach or exceed sample count)
- When data exhibits multicollinearity (highly correlated features)
- When interpretability is important (sparse coefficients are easier to interpret)
- When exploring solutions at different regularization strengths (the Homotopy method efficiently builds the entire solution path)

The main advantage of the Homotopy method over other LASSO solvers is that it computes the entire regularization path in one go, rather than requiring separate solutions for each λ value.

### How did you test your model to determine if it is working reasonably correctly?

I tested the model's correctness through multiple approaches:

1. **Functional Validation Tests**:
   - Verified model convergence on basic datasets
   - Checked prediction shape and reasonableness
   - Validated that coefficient estimates aren't all zero

2. **Sparsity Tests**:
   - Tested feature selection capability using collinear data
   - Verified that only a subset of features are selected from highly correlated groups
   - Checked zero coefficient counts at different regularization strengths

3. **Path Computation Tests**:
   - Verified that the λ path is decreasing
   - Confirmed that the coefficient path is piecewise linear
   - Checked that active set changes are as expected

4. **Numerical Stability Tests**:
   - Tested with features of different scales
   - Evaluated performance under extreme data conditions
   - Compared results with and without standardization

5. **Performance Comparisons**:
   - Calculated MSE and R² scores
   - Compared against simple benchmarks (like mean predictor)
   - For synthetic data, compared estimated coefficients to true coefficients

Test results confirm that the implementation produces sparse solutions across various data conditions and effectively handles collinearity.

### What parameters have you exposed to users of your implementation in order to tune performance?

I've exposed several key parameters that allow users to tune the model according to their needs:

1. **`max_iter`** (default: 1000)
   - Controls the maximum number of iterations for the homotopy path
   - May need to be increased for complex datasets, or decreased for simpler ones to save computation

2. **`tol`** (default: 1e-6)
   - Controls numerical precision and convergence criteria
   - Smaller values provide more precise results but may increase computational cost

3. **`lambda_max`** (optional)
   - Controls the starting point of the regularization path
   - Automatically calculated as the maximum correlation between features and target by default
   - Can be manually set by expert users to control the solution range

4. **`lambda_min_ratio`** (default: 1e-6)
   - Determines the endpoint of the regularization path (λ_min/λ_max)
   - Smaller values yield a more complete path but with higher computation cost
   - Larger values speed up computation but may miss solutions in low regularization regions

5. **`standardize`** (default: False)
   - Whether to automatically standardize features before fitting
   - Important for features of different scales to ensure fair penalization
   - When enabled, coefficients are automatically transformed back to original scale

These parameters allow users to balance computational efficiency and result precision.

### Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

The implementation does face several challenges, most of which could be addressed with further improvements:

1. **Computational Efficiency for High-Dimensional Data**
   - When feature count is very large (>10,000), matrix computations become slow and memory-intensive
   - Solution: Implement sparse matrix support, optimize matrix calculations, consider coordinate descent alternatives
   - This is a solvable engineering problem, not a fundamental algorithm limitation

2. **Numerical Stability Issues**
   - With highly collinear features, the Gram matrix approaches singularity, causing instability
   - Current implementation partly addresses this by regularizing the Gram matrix and using pseudoinverse
   - Further improvement: Implement more robust matrix decomposition methods like QR decomposition
   - This can be significantly improved, though collinearity itself presents challenges

3. **Memory Limitations**
   - Current implementation stores the entire dataset and computation results in memory
   - For extremely large datasets, this may be impractical
   - Potential solution: Implement chunk-based or online learning variants
   - This requires restructuring algorithm flow but is technically feasible

4. **Inherent LASSO Limitations**
   - LASSO may randomly select one variable from a group when group variables are highly correlated
   - Standardization option partially mitigates this issue
   - This is an inherent property of L1 regularization, but can be improved with extensions like Elastic Net


