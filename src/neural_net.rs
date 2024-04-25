// src/neural_net.rs

use crate::matrix::Matrix;

// Define the activation function you'll use
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

// Structure for a single layer in the neural network
pub struct Layer {
    weights: Matrix,
    biases: Vec<f64>,
    outputs: Vec<f64>,
    inputs: Vec<f64>,
    deltas: Vec<f64>, // For storing the error during backpropagation
}

impl Layer {
    // Initialize a layer with random weights and biases
    pub fn new(input_size: usize, output_size: usize) -> Layer {
        let weights = Matrix::new(vec![0.0; input_size * output_size], input_size, output_size);
        let biases = vec![0.0; output_size];
        let outputs = vec![0.0; output_size];
        let inputs = vec![0.0; input_size];
        let deltas = vec![0.0; output_size];
        Layer {
            weights,
            biases,
            outputs,
            inputs,
            deltas,
        }
    }

    // Compute the output of this layer given inputs
    pub fn forward(&mut self, inputs: Vec<f64>) {
        self.inputs = inputs;
        let mut results = self.weights.multiply_vec(&self.inputs);
        for i in 0..results.len() {
            results[i] += self.biases[i];
            self.outputs[i] = sigmoid(results[i]);
        }
    }

    // Calculate output error (delta)
    pub fn calculate_output_gradient(&mut self, target: &Vec<f64>) {
        self.deltas = self
            .outputs
            .iter()
            .zip(target.iter())
            .map(|(&output, &target)| sigmoid_derivative(output) * (output - target))
            .collect();
    }

    // Getter for outputs
    pub fn outputs(&self) -> &Vec<f64> {
        &self.outputs
    }
}

// Structure for the neural network
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new() -> NeuralNetwork {
        let input_layer = Layer::new(3, 4); // Example sizes: 3 inputs, 4 neurons in hidden layer
        let output_layer = Layer::new(4, 1); // 4 inputs from hidden layer, 1 output
        NeuralNetwork {
            layers: vec![input_layer, output_layer],
        }
    }

    // Perform a forward pass through the network
    pub fn forward(&mut self, inputs: Vec<f64>) {
        let mut inputs = inputs;
        for layer in &mut self.layers {
            layer.forward(inputs);
            inputs = layer.outputs().clone(); // Use the getter here
        }
    }

    // Perform backward propagation
    pub fn backward(&mut self, targets: &Vec<f64>) {
        let last_index = self.layers.len() - 1;
        self.layers[last_index].calculate_output_gradient(targets);

        for i in (0..last_index).rev() {
            let next_layer = &self.layers[i + 1];
            let current_layer = &mut self.layers[i];

            current_layer.deltas = current_layer
                .outputs()
                .iter()
                .enumerate()
                .map(|(index, &output)| {
                    let error_sum: f64 = next_layer
                        .weights
                        .data()
                        .iter()
                        .skip(index)
                        .step_by(next_layer.outputs().len())
                        .zip(next_layer.deltas.iter())
                        .map(|(&weight, &delta)| weight * delta)
                        .sum();
                    sigmoid_derivative(output) * error_sum
                })
                .collect();
        }
    }

    // Update weights and biases across all layers
    pub fn update_weights_and_biases(&mut self, learning_rate: f64) {
        for layer in &mut self.layers {
            for i in 0..layer.weights.rows() {
                for j in 0..layer.weights.cols() {
                    let weight_gradient = layer.inputs[j] * layer.deltas[i];
                    layer.weights.data_mut()[i * layer.weights.cols() + j] -=
                        learning_rate * weight_gradient;
                }
                layer.biases[i] -= learning_rate * layer.deltas[i];
            }
        }
    }

    // Training method combining forward, backward, and update steps
    pub fn train(&mut self, data: &[(Vec<f64>, Vec<f64>)], epochs: usize, learning_rate: f64) {
        for _ in 0..epochs {
            for &(ref inputs, ref outputs) in data {
                self.forward(inputs.clone());
                self.backward(outputs);
                self.update_weights_and_biases(learning_rate);
            }
        }
    }

    // Getter for layers
    pub fn layers(&self) -> &Vec<Layer> {
        &self.layers
    }
}
