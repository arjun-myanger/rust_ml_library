// Import DMatrix and DVector from the nalgebra crate for matrix and vector operations.
use nalgebra::{DMatrix, DVector};

// Define a structure for performing linear regression.
pub struct LinearRegression {
    // A vector of f64 that will hold the coefficients of the regression model.
    coefficients: DVector<f64>,
}

// Implementation block for LinearRegression.
impl LinearRegression {
    // Constructor method to create a new LinearRegression instance.
    pub fn new() -> Self {
        LinearRegression {
            // Initializes the coefficients vector to zero size, it will be set during fitting.
            coefficients: DVector::from_element(0, 0.0),
        }
    }

    // Method to fit the linear regression model to provided data.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        // Convert the input data (nested Vec<f64>) to a DMatrix for matrix operations.
        let x_matrix = DMatrix::from_row_slice(
            x.len(),                                               // number of rows
            x[0].len(),                                            // number of columns
            &x.iter().flat_map(|r| r.clone()).collect::<Vec<_>>(), // flatten the input 2D vector into a 1D vector
        );
        // Convert the target variable y into a DVector.
        let y_vector = DVector::from_column_slice(y);

        // Transpose the X matrix.
        let xt = x_matrix.transpose();
        // Multiply the transpose of X with X.
        let xt_x = &xt * &x_matrix;
        // Multiply the transpose of X with y.
        let xt_y = xt * y_vector;

        // Try to compute the inverse of X^T * X and then multiply it by X^T * y to find the coefficients.
        if let Some(xt_x_inv) = xt_x.try_inverse() {
            self.coefficients = xt_x_inv * xt_y;
        } else {
            // Print an error message if the matrix is not invertible.
            eprintln!("Matrix is not invertible.");
        }
    }

    // Method to make predictions using the fitted model.
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        // Convert the input x for prediction into a DMatrix like in the fit method.
        let x_matrix = DMatrix::from_row_slice(
            x.len(),
            x[0].len(),
            &x.iter().flat_map(|r| r.clone()).collect::<Vec<_>>(),
        );
        // Multiply the input matrix by the coefficients to get predictions.
        let result = x_matrix * &self.coefficients;
        // Extract the first column of the resulting matrix as the prediction result vector.
        result.column(0).iter().copied().collect()
    }
}
