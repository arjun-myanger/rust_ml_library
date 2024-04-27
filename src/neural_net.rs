// src/neural_net.rs

use crate::matrix::Matrix;

// Define the activation function you'll use
pub fn sigmoid(x: f64) -> f64 {
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
        let input_layer = Layer::new(2, 4); // Adjusted to 2 inputs to match the XOR data
        let output_layer = Layer::new(4, 1);
        NeuralNetwork {
            layers: vec![input_layer, output_layer],
        }
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
// Perform backward propagation
pub fn backward(&mut self, targets: &Vec<f64>) {
    let last_index = self.layers.len() - 1;
    self.layers[last_index].calculate_output_gradient(targets);

    // First, copy all necessary data from each layer into temporary structures
    let layer_data: Vec<_> = self
        .layers
        .iter()
        .map(|layer| {
            (
                layer.weights.data().clone(),
                layer.outputs().clone(),
                layer.deltas.clone(),
            )
        })
        .collect();

    // Now, we use the copied data to avoid borrowing `self.layers` while it's being mutated
    for i in (0..last_index).rev() {
        let (weights_next, outputs_next, deltas_next) = &layer_data[i + 1];
        let current_layer = &mut self.layers[i];

        current_layer.deltas = current_layer
            .outputs()
            .iter()
            .enumerate()
            .map(|(index, &output)| {
                let error_sum: f64 = weights_next
                    .iter()
                    .skip(index)
                    .step_by(outputs_next.len())
                    .zip(deltas_next)
                    .map(|(&weight, &delta)| weight * delta)
                    .sum();
                sigmoid_derivative(output) * error_sum
            })
            .collect();
    }
}

// Note that the above code involves cloning the weights, outputs, and deltas for each layer
// which could lead to increased memory usage. This is generally acceptable for small to moderate sized networks
// but for very large models, more memory-efficient approaches may be necessary.

// Update weights and biases across all layers
// Update weights and biases across all layers
pub fn update_weights_and_biases(&mut self, learning_rate: f64) {
    // Iterate over each layer and update weights and biases
    for layer in &mut self.layers {
        // Calculate the weight gradients first
        let mut weight_gradients = vec![vec![0.0; layer.weights.cols()]; layer.weights.rows()];
        for i in 0..layer.weights.rows() {
            for j in 0..layer.weights.cols() {
                weight_gradients[i][j] = layer.inputs[j] * layer.deltas[i] * learning_rate;
            }
        }

        // Now, apply the weight gradients
        for i in 0..layer.weights.rows() {
            for j in 0..layer.weights.cols() {
                // Directly access data_mut to update weights
                let data_index = i * layer.weights.cols() + j;
                *layer.weights.data_mut().get_mut(data_index).unwrap() -= weight_gradients[i][j];
            }
            // Update biases in the same loop
            layer.biases[i] -= layer.deltas[i] * learning_rate;
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
