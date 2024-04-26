// lib.rs

pub mod matrix;
pub mod neural_net;
pub mod regression;

// Re-export necessary structs and functions from neural_net and matrix
pub use matrix::Matrix;
pub use neural_net::{Layer, NeuralNetwork}; // Assuming Layer and NeuralNetwork are defined in neural_net // Re-export Matrix for external use

// Optionally, if sigmoid is defined in neural_net and needs to be accessible
// This assumes `sigmoid` is a public function defined in the neural_net module.
// If not, you'll need to add `pub` before `fn sigmoid` in its definition.
pub use neural_net::sigmoid;
