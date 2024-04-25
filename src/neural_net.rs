// src/neural_net.rs

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
            inputs = layer.outputs.clone();
        }
    }
}
