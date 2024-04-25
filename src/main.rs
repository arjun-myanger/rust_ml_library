// main.rs

// Assuming your library is named `rust_ml_library` and is correctly referenced in your Cargo.toml
extern crate rust_ml_library;
use rust_ml_library::neural_net::NeuralNetwork;

fn main() {
    // Training data for the XOR function
    let train_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    // Initialize the neural network
    let mut network = NeuralNetwork::new();

    // Train the network with 1000 epochs and a learning rate of 0.1
    network.train(&train_data, 1000, 0.1);

    // Test the network
    for (inputs, expected) in train_data.iter() {
        network.forward(inputs.clone());
        // Using the getter to access outputs of the last layer
        let outputs = network.layers().last().unwrap().outputs();
        println!(
            "Input: {:?}, Expected: {:?}, Output: {:?}",
            inputs, expected, outputs
        );
    }
}
