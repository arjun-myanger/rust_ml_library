// src/matrix.rs
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data does not match matrix dimensions"
        );
        Matrix { data, rows, cols }
    }

    // Getter for data
    pub fn data(&self) -> &Vec<f64> {
        &self.data
    }

    // You might also want to add getters for rows and cols if needed
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    // Example method for matrix addition
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Row mismatch");
        assert_eq!(self.cols, other.cols, "Column mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Matrix::new(data, self.rows, self.cols)
    }
}
