use ndarray::{arr1, arr2, Array1, Array2};
use ndarray_linalg::Inverse;

pub struct LinearRegression {
    coefficients: Vec<f64>,
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression {
            coefficients: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        // Convert vectors to ndarray structures
        let x_matrix = Array2::from_shape_vec(
            (x.len(), x[0].len()),
            x.iter().flat_map(|r| r.iter().cloned()).collect(),
        )
        .unwrap();
        let y_vector = Array1::from_vec(y.to_vec());

        // Calculate (X^T * X)
        let xt_x = x_matrix.t().dot(&x_matrix);

        // Calculate the inverse of (X^T * X)
        let xt_x_inv = xt_x.inv().expect("Matrix is not invertible");

        // Calculate (X^T * Y)
        let xt_y = x_matrix.t().dot(&y_vector);

        // Calculate the coefficients (beta) = (X^T * X)^(-1) * (X^T * Y)
        let beta = xt_x_inv.dot(&xt_y);

        // Store the coefficients
        self.coefficients = beta.to_vec();
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                row.iter()
                    .zip(&self.coefficients)
                    .map(|(xi, &bi)| xi * bi)
                    .sum()
            })
            .collect()
    }
}
