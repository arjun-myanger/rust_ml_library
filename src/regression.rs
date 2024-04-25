use nalgebra::{DMatrix, DVector};

pub struct LinearRegression {
    coefficients: DVector<f64>,
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression {
            coefficients: DVector::from_element(0, 0.0), // Initialize with zero-sized vector
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        // Convert input data to DMatrix and DVector
        let x_matrix = DMatrix::from_row_slice(
            x.len(),
            x[0].len(),
            &x.iter().flat_map(|r| r.clone()).collect::<Vec<_>>(),
        );
        let y_vector = DVector::from_column_slice(y);

        // Perform the matrix calculations
        let xt = x_matrix.transpose();
        let xt_x = &xt * &x_matrix;
        let xt_y = xt * y_vector;

        // Calculate the coefficients
        if let Some(xt_x_inv) = xt_x.try_inverse() {
            self.coefficients = xt_x_inv * xt_y;
        } else {
            // Handle the case where the matrix is not invertible
            eprintln!("Matrix is not invertible.");
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let x_matrix = DMatrix::from_row_slice(
            x.len(),
            x[0].len(),
            &x.iter().flat_map(|r| r.clone()).collect::<Vec<_>>(),
        );
        let result = x_matrix * &self.coefficients;
        result.column(0).iter().copied().collect()
    }
}
