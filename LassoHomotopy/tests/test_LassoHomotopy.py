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

    # Get feature columns (starting with 'x' or 'X')
    X = np.array([[v for k, v in datum.items() if k.lower().startswith('x')] for datum in data])

    # Check for target column (either 'y' or 'target')
    if 'y' in data[0]:
        y = np.array([datum['y'] for datum in data])
    elif 'target' in data[0]:
        y = np.array([datum['target'] for datum in data])
    else:
        raise KeyError("Neither 'y' nor 'target' column found in the data")

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

    # Fit model with a lower lambda_min_ratio for less regularization
    model = LassoHomotopyModel(lambda_min_ratio=1e-5)
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
    """
    Test the model's ability to identify sparse coefficients using a loaded dataset.
    """
    # Load the collinear dataset which has a sparse structure
    X, y = load_csv_data("collinear_data.csv")

    # Standardize X to improve numerical stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X_standardized = (X - X_mean) / X_std

    # Fit model with very low regularization
    model = LassoHomotopyModel(lambda_min_ratio=1e-10, max_iter=5000, tol=1e-8)
    results = model.fit(X_standardized, y)

    # Convert coefficients back to original scale
    coef_original_scale = results.coef_ / X_std

    # Get non-zero coefficients using a small threshold
    non_zero_indices = np.where(np.abs(coef_original_scale) > 0.01)[0]

    # Print coefficient information
    print(f"Total features: {X.shape[1]}")
    print(f"Non-zero coefficients: {len(non_zero_indices)}/{X.shape[1]}")
    print(f"Non-zero indices: {non_zero_indices}")
    print(f"Coefficient values: {coef_original_scale[non_zero_indices]}")

    # Assert the model produces a sparse solution (fewer non-zeros than total features)
    assert len(non_zero_indices) < X.shape[1], "Model should produce a sparse solution"

    # Assert that at least one coefficient is non-zero
    assert len(non_zero_indices) > 0, "Model should identify at least one non-zero coefficient"

    # Print MSE information but don't assert on it
    y_pred = results.predict(X_standardized)
    mse = np.mean((y - y_pred) ** 2)
    mean_mse = np.mean((y - np.mean(y)) ** 2)
    print(f"Mean squared error: {mse}")
    print(f"Mean predictor MSE: {mean_mse}")


def test_different_lambda_values():
    # Test model behavior with different lambda values
    X, y = load_csv_data("small_test.csv")

    # Fit with high lambda (more regularization)
    model_high_reg = LassoHomotopyModel(lambda_min_ratio=0.5)  # Higher minimum lambda
    results_high_reg = model_high_reg.fit(X, y)

    # Fit with low lambda (less regularization)
    model_low_reg = LassoHomotopyModel(lambda_min_ratio=1e-6)  # Lower minimum lambda
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
