import numpy as np
import scipy.special

# Neural network class definition
class NeuralNetwork:

    # Initialize the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Set number of nodes in each input, hidden, and output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Learning rate
        self.lr = learningrate

        # Link weight matrices: wih (input to hidden) and who (hidden to output)
        # Weights are sampled from a normal distribution centered at 0.0
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # Activation function (sigmoid function)
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # Train the neural network
    def train(self, inputs_list, targets_list):
        # Convert inputs and targets lists to 2D arrays
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Output layer error is (target - actual)
        output_errors = targets - final_outputs
        # Hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    # Query the neural network
    def query(self, inputs_list):
        # Convert inputs list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs