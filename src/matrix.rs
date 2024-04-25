// Define a Matrix struct that will store the elements of the matrix, the number of rows, and the number of columns.
pub struct Matrix {
    data: Vec<f64>, // Vector that holds the matrix elements in a flat format.
    rows: usize,    // The number of rows in the matrix.
    cols: usize,    // The number of columns in the matrix.
}

// Implement methods for the Matrix struct.
impl Matrix {
    // Constructor for creating a new Matrix. Requires the matrix data, and the dimensions (rows and columns).
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        // Ensure that the length of the data vector matches the product of rows and columns.
        assert_eq!(
            data.len(),
            rows * cols,
            "Data does not match matrix dimensions"
        );
        // Create a new Matrix instance with the provided data and dimensions.
        Matrix { data, rows, cols }
    }

    // Getter method for the matrix data.
    pub fn data(&self) -> &Vec<f64> {
        // Returns a reference to the matrix data.
        &self.data
    }

    // Getter method for the number of rows in the matrix.
    pub fn rows(&self) -> usize {
        // Returns the number of rows.
        self.rows
    }

    // Getter method for the number of columns in the matrix.
    pub fn cols(&self) -> usize {
        // Returns the number of columns.
        self.cols
    }

    // Method to add this matrix with another matrix, returning a new matrix as the result.
    pub fn add(&self, other: &Matrix) -> Matrix {
        // Ensure the two matrices have the same dimensions.
        assert_eq!(self.rows, other.rows, "Row mismatch");
        assert_eq!(self.cols, other.cols, "Column mismatch");
        // Create a new vector by adding corresponding elements of the two matrices.
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        // Return a new Matrix instance with the summed data and same dimensions.
        Matrix::new(data, self.rows, self.cols)
    }

    // Method to multiply this matrix by a vector, returning the resulting vector.
    pub fn multiply_vec(&self, vec: &Vec<f64>) -> Vec<f64> {
        // Ensure that the number of columns in the matrix matches the length of the vector.
        assert_eq!(self.cols, vec.len(), "Dimension mismatch");
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.data[i * self.cols + j] * vec[j];
            }
        }
        result
    }
}
