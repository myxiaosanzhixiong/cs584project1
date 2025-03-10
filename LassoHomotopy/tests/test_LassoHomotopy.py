import csv
import os
import numpy as np
import pytest

from model.LassoHomotopy import LassoHomotopyModel

def load_csv_data(filename):
    data = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append({k: float(v) for k, v in row.items()})
    
    X = np.array([[v for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([datum['y'] for datum in data])
    return X, y

def test_small_dataset():
    # Load the small test dataset
    X, y = load_csv_data("small_test.csv")
    
    # Fit model
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    
    # Check that coefficients are not all zero
    assert np.any(results.coef_ != 0), "Model should learn non-zero coefficients"
    
    # Check prediction
    preds = results.predict(X)
    assert preds.shape == y.shape, "Predictions should have same shape as y"

def test_collinear_data():
    # Load the collinear dataset
    X, y = load_csv_data("collinear_data.csv")
    
    # Fit model
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    
    # In collinear data, LASSO should produce sparse solutions
    # Check that many coefficients are exactly zero
    zero_coefs = np.sum(np.abs(results.coef_) < 1e-10)
    non_zero_coefs = len(results.coef_) - zero_coefs
    
    # Assert that we have a sparse solution (at least some zeros)
    assert zero_coefs > 0, "LASSO should produce sparse solutions with collinear data"
    
    # Print the sparsity for informational purposes
    print(f"Sparsity: {zero_coefs}/{len(results.coef_)} coefficients are zero")

def test_synthetic_data():
    # Create synthetic data with known coefficients
    np.random.seed(42)
    n_samples, n_features = 100, 20
    
    # True coefficients: only 5 out of 20 are non-zero
    true_coef = np.zeros(n_features)
    true_coef[:5] = np.array([3.5, -2.0, 1.5, -1.0, 0.5])
    
    # Create design matrix and response with noise
    X = np.random.randn(n_samples, n_features)
    y = X @ true_coef + np.random.normal(0, 0.5, n_samples)
    
    # Fit model
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    
    # The model should identify the non-zero coefficients
    # and set others close to zero
    non_zero_indices = np.where(np.abs(results.coef_) > 0.1)[0]
    
    # Check if most of the non-zero coefficients are in the first 5 indices
    correct_non_zeros = sum(idx < 5 for idx in non_zero_indices)
    
    assert correct_non_zeros >= 3, "Model should identify most of the true non-zero coefficients"
    
    # Print info for debugging
    print(f"True non-zero coefficients: {true_coef[:5]}")
    print(f"Estimated coefficients: {results.coef_[:5]}")
    print(f"Number of correctly identified non-zero coefficients: {correct_non_zeros}/5")

def test_different_lambda_values():
    # Test model behavior with different lambda values
    X, y = load_csv_data("small_test.csv")
    
    # Fit with high lambda (more regularization)
    model_high_reg = LassoHomotopyModel(lambda_min_ratio=0.5)  # Higher minimum lambda
    results_high_reg = model_high_reg.fit(X, y)
    
    # Fit with low lambda (less regularization)
    model_low_reg = LassoHomotopyModel(lambda_min_ratio=1e-4)  # Lower minimum lambda
    results_low_reg = model_low_reg.fit(X, y)
    
    # Higher regularization should result in more zero coefficients
    high_reg_zeros = np.sum(np.abs(results_high_reg.coef_) < 1e-10)
    low_reg_zeros = np.sum(np.abs(results_low_reg.coef_) < 1e-10)
    
    print(f"High regularization zeros: {high_reg_zeros}/{len(results_high_reg.coef_)}")
    print(f"Low regularization zeros: {low_reg_zeros}/{len(results_low_reg.coef_)}")
    
    # The model should produce valid predictions
    preds_high = results_high_reg.predict(X)
    preds_low = results_low_reg.predict(X)
    
    assert preds_high.shape == y.shape
    assert preds_low.shape == y.shape
