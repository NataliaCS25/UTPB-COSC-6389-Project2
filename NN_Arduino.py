import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle, resample
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from tkinter import Scale
from tkinter.ttk import Progressbar
from collections import Counter
from tkinter import messagebox
import os

# Classes dictionary
class_to_action = {
    0: "Turn Left",
    1: "Move Backward",
    2: "Move Forward",
    3: "Turn Right"
}

trained = False
nn= None

# NN configuration
hidden_sizes = [16, 8]
num_epochs = 50000

# Sensor input limits
LIMITS = {
    "Distance Sensor": (-1, 1),  # Distance sensor
    "Obstacle Position": (-1, 1)  # Obstacle position
}

# NN class
class ArduinoNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.00001, momentum=0.9, regularization=0.0001, activation_function="ReLU"):
        # Initialize the neural network with input size, hidden layers, and output size
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            # He initialization for weights
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]))  
            self.biases.append(np.zeros(layer_sizes[i + 1]))

        # Initialize velocity for momentum optimization
        self.velocities = [np.zeros_like(w) for w in self.weights]
    
    def activate(self, x):
        if self.activation_function == "ReLU":
            return np.maximum(0, x)
        elif self.activation_function == "Sigmoid":
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "Tanh":
            return np.tanh(x)
        elif self.activation_function== "Softmax":
            exps = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")
    
    def activate_derivative(self, x):
        if self.activation_function == "ReLU":
            return np.where(x > 0, 1, 0)
        elif self.activation_function == "Sigmoid":
            s = self.activate(x)
            return s * (1 - s)
        elif self.activation_function == "Tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation_function == "Softmax":
            return self.softmax_derivative(self.activate(x))
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")

    def softmax_derivative(self,softmax_output):
        """
        Computes the derivative of the softmax function for backpropagation.
        Args: softmax_output (numpy.ndarray): The output of the softmax function (batch_size, num_classes).
        Returns: numpy.ndarray: The gradient of the softmax output (same shape as softmax_output).
        """
        return softmax_output * (1 - softmax_output)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        # Softmax activation for output layer
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def compute_loss(self, expected_output):
        # Compute cross-entropy loss
        predictions = self.layer_outputs[-1]
        predictions = np.clip(predictions, 1e-12, 1 - 1e-12)  # Avoid extreme values
        return -np.mean(np.sum(expected_output * np.log(predictions), axis=1))

    def forward_propagation(self, inputs):
        # Forward propagation through the network
        self.layer_inputs = []
        self.layer_outputs = [inputs]

        for w, b in zip(self.weights[:-1], self.biases[:-1]):  # For hidden layers
            layer_input = np.dot(self.layer_outputs[-1], w) + b
            self.layer_inputs.append(layer_input)
            self.layer_outputs.append(self.activate(layer_input))

        # Apply softmax for output layer
        # self.layer_outputs[-1] = self.softmax(self.layer_inputs[-1])
        layer_input = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
        self.layer_inputs.append(layer_input)
        self.layer_outputs.append(self.softmax(layer_input))

        return self.layer_outputs[-1]

    def backward_propagation(self, inputs, expected_output):
        # Backpropagation to update weights and biases
        output_error = self.layer_outputs[-1] - expected_output
        deltas = [output_error]

        for i in reversed(range(len(self.weights) - 1)):
            error = np.dot(deltas[-1], self.weights[i + 1].T)
            delta = error * self.activate_derivative(self.layer_inputs[i])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            gradient = np.dot(self.layer_outputs[i].T, deltas[i])
            gradient = np.clip(gradient, -0.5, 0.5)  # Clipping
            self.weights[i] -= self.learning_rate * gradient
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0)

    def train(self, inputs, expected_output, epochs, on_epoch_end=None):
        # Training
        losses = []
        for epoch in range(epochs):
            self.forward_propagation(inputs)
            self.backward_propagation(inputs, expected_output)
            loss = self.compute_loss(expected_output)
            losses.append(loss)

            # Update logs 
            if epoch % 100 == 0 or epoch == epochs - 1:
                loss = self.compute_loss(expected_output)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

                if on_epoch_end:
                    on_epoch_end(epoch, loss, self.weights)

        return losses

    def predict(self, inputs):
        # Predictions using the trained network
        return self.forward_propagation(inputs)

# Netwrok architecture
def show_nn_structure():
    structure_window = tk.Toplevel(root)
    structure_window.title("Netwrok Architecture")

    # Show layers and neurons
    layer_info = f"Input layer:\nInput: {inputs_train_balanced.shape[1]} neurons\n"
    for i, layer_size in enumerate(hidden_sizes):
        layer_info += f"Hidden layer {i+1}: {layer_size} neurons\n"
    layer_info += f"Output: {outputs_train_balanced_onehot.shape[1]} neurons\n"

    tk.Label(structure_window, text=layer_info, font=("Arial", 12)).pack(pady=10)

def draw_nn(canvas_nn, predicted_class = None, weights=None):
    # Draw Network on canvas
    canvas_nn.delete("all")
    layer_positions = []  # Save positions per layer

    # NN layers
    layer_sizes = [2] + hidden_sizes + [outputs_train_balanced_onehot.shape[1]]  # 2 input, hidden layers, 4 outputs
    canvas_width = canvas_nn.winfo_width()
    canvas_height = canvas_nn.winfo_height()

    # Horizontal space between layers
    x_spacing = canvas_width / (len(layer_sizes) + 1)

    for i, layer_size in enumerate(layer_sizes):
        y_spacing = canvas_height / (layer_size + 1)
        layer_positions.append([])
        for j in range(layer_size):
            x = (i + 1) * x_spacing
            y = (j + 1) * y_spacing
            layer_positions[-1].append((x, y))
            canvas_nn.create_oval(x - 10, y - 10, x + 10, y + 10, fill="lightgray", outline="black")

            # Highlight output
            if i == len(layer_sizes) - 1 and predicted_class is not None and j == predicted_class:
                canvas_nn.create_oval(x - 12, y - 12, x + 12, y + 12, fill="lightgreen", outline="black")

    # Draw axon
    max_weight = max(abs(w).max() for w in weights) if weights else 1  # Normalize weights
    min_weight = min(abs(w).min() for w in weights) if weights else 0.01

    for i in range(len(layer_positions) - 1):
        for j, start in enumerate(layer_positions[i]):
            for k, end in enumerate(layer_positions[i + 1]):
                color = "lightgray"
                width=1
                if weights is not None and i < len(weights):
                    weight = weights[i][j, k]
                    # Normalize weights
                    normalized_weight = (abs(weight) - min_weight) / (max_weight - min_weight + 1e-5)
                    if weight > 0:
                        color = f"#{int(255 * normalized_weight):02x}0000"  # Red
                    elif weight < 0:
                        color = f"#0000{int(255 * normalized_weight):02x}"  # Blue

                    width = int(1 + 4 * normalized_weight)  # Dynamic width
                    
                # Highlight axons for predicted class
                if i == len(layer_positions) - 2 and predicted_class is not None:
                    if k == predicted_class:
                        color = "red" 
                        width = 3
                    else:
                        color = "lightgray"
                        width = 1
                    
                canvas_nn.create_line(start[0], start[1], end[0], end[1], fill=color,width=width*0.5)

def draw_car(canvas_car, action):
    # Draw the car and indicate the action
    canvas_car.delete("all")
    canvas_width = canvas_car.winfo_width()
    canvas_height = canvas_car.winfo_height()

    car_x = canvas_width / 2
    car_y = canvas_height / 2
    canvas_car.create_rectangle(car_x - 40, car_y - 20, car_x + 40, car_y + 20, fill="blue")
    canvas_car.create_text(250, 15, text="Car Action", font=("Arial", 12), fill="black")

    # Draw the car's direction based on the action
    if action == "Move Forward":
        canvas_car.create_text(car_x, car_y - 40, text="↑", font=("Arial", 24), fill="green")
    elif action == "Move Backward":
        canvas_car.create_text(car_x, car_y + 40, text="↓", font=("Arial", 24), fill="red")
    elif action == "Turn Left":
        canvas_car.create_text(car_x - 50, car_y, text="←", font=("Arial", 24), fill="orange")
    elif action == "Turn Right":
        canvas_car.create_text(car_x + 50, car_y, text="→", font=("Arial", 24), fill="purple")

def train_nn():
    # Trains the NN and updates the UI
    global nn, trained
    
    result_text.set("Training the network...")
    root.update()

    # Retrieve the selected activation function
    activation_function = selected_activation.get()

    # Create NN
    nn = ArduinoNeuralNetwork(input_size=inputs_train_balanced.shape[1], hidden_sizes=hidden_sizes,
                              output_size=outputs_train_balanced_onehot.shape[1],activation_function=activation_function)
    
    def on_epoch_end(epoch, loss, weights):
        # Callback executed at the end of each epoch to update the UI
        result_text.set(f"Epoch: {epoch + 1}, Loss: {loss:.4f}")
        update_loss_graph(epoch, loss)  # Update loss graph
        draw_nn(canvas_nn, predicted_class=None, weights=weights) #update NN visualization
        update_progress(epoch, num_epochs) #Update training progress bar
        root.update()

    # Train the NN
    nn.train(inputs_train_balanced, outputs_train_balanced_onehot, epochs=num_epochs, on_epoch_end=on_epoch_end)

    trained = True
    result_text.set("Neural Network succesfully trained! \nReady for predictions!")

def predict_action():
    global nn
    # Makes a prediction if the NN is trained
    if not trained or nn is None:
        result_text.set("Please train the NN before making predictions")
        return

    try:
        # Get the sensor values from the sliders
        sensor_distance = slider_distance.get()
        obstacle_position = slider_obstacle.get()

        # Normalize the input values
        inputs = np.array([[sensor_distance, obstacle_position]])
        inputs = (inputs - np.mean(inputs_train_balanced, axis=0)) / np.std(inputs_train_balanced, axis=0)

        # Perform the prediction
        predictions = nn.predict(inputs)
        predicted_class = np.argmax(predictions) # Get the class with the highest probability
        action = class_to_action[predicted_class] # Map the class to the corresponding action

        # print(f"Precitions: {predictions}")
        # print(f"Class: {predicted_class}")

        probabilities_percentage = [f"{prob * 100:.2f}%" for prob in predictions[0]]

        result_text.set(f"Action: {action}\n\nPredicted Class: {predicted_class}\n\nProbabilities: {probabilities_percentage}")

        draw_car(canvas_car, action) # Visualize car action
        draw_nn(canvas_nn, predicted_class=predicted_class,weights=nn.weights) # Visualize NN

    except ValueError:
        result_text.set("Error processing the values. Please try again")

def update_loss_graph(epoch, loss):
    # Update loss graph on canvas
    max_loss = 2.0 # max for scaling
    min_loss = 0.0
    global loss_points

    if epoch == 0:
        # Clear the canvas at the start
        canvas_loss.delete("all")

        # Draw the graph border
        canvas_loss.create_rectangle(40, 10, 490, 190, outline="black")

        # Add axis labels
        canvas_loss.create_text(35, 10, text=f"{max_loss:.1f}", anchor="e", font=("Arial", 10))  # Max loss
        canvas_loss.create_text(35, 190, text=f"{min_loss:.1f}", anchor="e", font=("Arial", 10))  # Min loss
        canvas_loss.create_text(265, 200, text="Epochs", anchor="center", font=("Arial", 12))  # X-axis label
        canvas_loss.create_text(20, 100, text="Loss", anchor="center", font=("Arial", 12), angle=90)  # Y-axis label

        loss_points = []  # Reset loss points list

    # Define the graph range
    max_epochs = num_epochs

    # Scale and position the new point
    x = 40 + (450 / max_epochs) * epoch  # Adjust X scaling to fit within the new border
    y = 190 - (170 * (loss - min_loss) / (max_loss - min_loss))  # Adjust Y scaling for padding
    loss_points.append((x, y))

    # Draw the loss line
    if len(loss_points) > 1:
        canvas_loss.create_line(loss_points[-2], loss_points[-1], fill="blue")

def update_epochs():
    # Updates number of epochs based on the user input
    global num_epochs
    try:
        epochs = int(entry_epochs.get())
        if epochs <= 0:
            raise ValueError("The number of epochs must be a positive integer.")
        num_epochs = epochs
        result_text.set(f"Number of epochs updated to {num_epochs}.")
    except ValueError:
        result_text.set("Please enter a valid number of epochs")
    return num_epochs

def validate_inputs(value, min_val, max_val):
    # Validate inputs
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value

def update_progress(epoch, total_epochs):
    # Update progress bar
    progress["value"] = (epoch / total_epochs) * 100
    root.update()

def show_info_message():
    messagebox.showinfo(
        title="About the Neural Network",
        message=(
            "Welcome to NeuroDrive Arduino!\n\n"
            "Objective:\n"
            "The goal of this neural network is to predict the action for an Arduino-controlled car.\n"
            "The network processes sensor inputs, such as distance and obstacle position, "
            "and predicts the appropriate action (e.g., move forward, move backward, turn left, turn right).\n\n"
            "How to Use:\n"
            "- Adjust sensor values using the sliders.\n"
            "- Configure the network using the controls provided.\n"
            "- Train the network and visualize its predictions.\n\n"
        )
    )

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

data = pd.read_csv("Arduino.csv")

# Extract inputs and outputs from the dataset
inputs = data[["Distance Sensor", "Obstacle"]].values
outputs = data[["Output: Turn", "Output: Direction"]].values

# Split data intro training and testing sets
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
    inputs, outputs, test_size=0.2, random_state=42)

# Oversampling for class balance
train_data = pd.DataFrame(inputs_train, columns=["Distance Sensor", "Obstacle"])
train_data["Output: Turn"] = outputs_train[:, 0]
train_data["Output: Direction"] = outputs_train[:, 1]

# Identify majority and minority classes
majority_class = train_data[(train_data["Output: Turn"] == 0) & (train_data["Output: Direction"] == 1)]
minority_classes = train_data[(train_data["Output: Turn"] != 0) | (train_data["Output: Direction"] != 1)]

minority_classes_upsampled = resample(minority_classes, replace=True, n_samples=len(majority_class), random_state=42)

# Combine majority and upsampled minority classes
balanced_train_data = pd.concat([majority_class, minority_classes_upsampled])

# Extract balanced inputs and outputs
inputs_train_balanced = balanced_train_data[["Distance Sensor", "Obstacle"]].values
outputs_train_balanced = balanced_train_data[["Output: Turn", "Output: Direction"]].values

# Map unique output combinations to classes
outputs_combined = np.unique(outputs_train_balanced, axis=0)
mapping = {tuple(row): i for i, row in enumerate(outputs_combined)}

# Convert outputs to class indices
outputs_train_balanced_mapped = np.array([mapping[tuple(row)] for row in outputs_train_balanced])
outputs_test_mapped = np.array([mapping[tuple(row)] for row in outputs_test])

# print("Class Mapping:")
# for key, value in mapping.items():
#     print(f"Class {value}: {key}")

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
outputs_train_balanced_onehot = encoder.fit_transform(outputs_train_balanced_mapped.reshape(-1, 1))

# Print mapped and encoded outputs
# print("Class distribution in training set:")
# print(np.unique(outputs_train_balanced_mapped, return_counts=True))
# print(outputs_train_balanced_onehot[:10])

# Normalize inputs
inputs_train_balanced = (inputs_train_balanced - np.mean(inputs_train_balanced, axis=0)) / np.std(inputs_train_balanced, axis=0)
inputs_test = (inputs_test - np.mean(inputs_train_balanced, axis=0)) / np.std(inputs_train_balanced, axis=0)

# Test accuracy
# accuracy = accuracy_score(true_classes, predicted_classes)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
# cm = confusion_matrix(true_classes, predicted_classes)
# class_counts = Counter(predicted_classes)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=mapping.keys(), yticklabels=mapping.keys())
# plt.xlabel("Predicted classes")
# plt.ylabel("Real classes")
# plt.title("Confusion matrix")
# plt.show()

root = tk.Tk()
root.title("NeuroDrive Arduino")
root.after(100, show_info_message)

# Progress bar
progress = Progressbar(root, orient="horizontal", length=200, mode="determinate")
progress.grid(row=4, column=1, padx=10, pady=10)

# Configuration frame
frame_config = ttk.Frame(root, padding="10")
frame_config.grid(row=0, column=0, sticky="W")
ttk.Label(frame_config, text="Number of Epochs:").grid(row=0, column=0, sticky="e")
entry_epochs = ttk.Entry(frame_config, width=10)
entry_epochs.grid(row=0, column=1)
entry_epochs.insert(0, "50000") 
ttk.Button(frame_config, text="Update Epochs", command=update_epochs).grid(row=0, column=2, padx=5)

frame_inputs = ttk.Frame(root, padding="10")
frame_inputs.grid(row=2, column=0, sticky="W")
ttk.Label(frame_inputs, text="Distance Sensor:").grid(row=0, column=0, sticky="W")
slider_distance = tk.Scale(frame_inputs, from_=LIMITS["Distance Sensor"][0], to=LIMITS["Distance Sensor"][1],
                            resolution=0.01, orient="horizontal", length=200)
slider_distance.grid(row=0, column=1)

ttk.Label(frame_inputs, text="Obstacle Position:").grid(row=1, column=0, sticky="W")
slider_obstacle = tk.Scale(frame_inputs, from_=LIMITS["Obstacle Position"][0], to=LIMITS["Obstacle Position"][1],
                           resolution=0.01, orient="horizontal", length=200)
slider_obstacle.grid(row=1, column=1)

ttk.Button(frame_inputs, text="Predict Action", command=predict_action).grid(row=3, column=0, padx=10)

frame_buttons = ttk.Frame(root, padding="10")
frame_buttons.grid(row=1, column=0, sticky="W")
ttk.Button(frame_buttons, text="Train Neural Network", command=train_nn).grid(row=0, column=0, padx=5)

# Add activation function selector
ttk.Label(frame_config, text="Activation Function:").grid(row=1, column=0, sticky="e")
activation_functions = ["ReLU", "Sigmoid", "Tanh", "Softmax"]  # Add more as needed
selected_activation = tk.StringVar(value="ReLU")  # Default value
activation_dropdown = ttk.Combobox(frame_config, textvariable=selected_activation, values=activation_functions, state="readonly")
activation_dropdown.grid(row=1, column=1)

# Results frame
frame_results = ttk.Frame(root, padding="10")
frame_results.grid(row=3, column=0, sticky="W")
result_text = tk.StringVar()
ttk.Label(frame_results, textvariable=result_text, font=("Arial", 12)).grid(row=3, column=0, sticky="W")

# Canvas for visualization
canvas_nn = tk.Canvas(root, width=500, height=350, bg="white")
canvas_nn.grid(row=0, column=1, rowspan=2, padx=10, pady=10)

# Car canvas
canvas_car = tk.Canvas(root, width=500, height=150, bg="white")
canvas_car.grid(row=3, column=1, padx=10, pady=10)

# Loss canvas
canvas_loss = tk.Canvas(root, width=500, height=215, bg="white")
canvas_loss.grid(row=2, column=1, padx=10, pady=10)

root.mainloop()