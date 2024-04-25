#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use rust_ml_library::regression::LinearRegression; // This crate provides methods to assert approximate equality

    #[test]
    fn test_linear_regression_fit_and_predict() {
        let mut model = LinearRegression::new();
        let x = vec![vec![1.0, 1.0], vec![1.0, 2.0], vec![1.0, 3.0]]; // Including intercept term
        let y = vec![1.0, 2.0, 3.0];
        model.fit(&x, &y);
        let predictions = model.predict(&x);
        let expected = vec![1.0, 2.0, 3.0]; // Simplified expectation for illustration

        // Using assert_abs_diff_eq to check if the values are approximately equal within a small epsilon
        for (pred, exp) in predictions.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(pred, exp, epsilon = 0.01);
        }
    }
}
