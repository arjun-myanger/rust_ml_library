// tests/matrix_tests.rs
use rust_ml_library::matrix::Matrix;

#[test]
fn test_matrix_addition() {
    let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Matrix::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let result = a.add(&b);
    assert_eq!(result.data(), &vec![6.0, 8.0, 10.0, 12.0]);
}
