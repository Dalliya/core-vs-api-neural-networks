import sys
import os
import numpy as np
from sklearn.metrics import confusion_matrix

# Adjust system path to allow imports from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.custom_network import NeuralNetwork
from utils.data_loader import load_fashion_mnist
from visualizations.plotter import generate_3d_topology

def main() -> None:
    # 1. Configuration & Hyperparameters (Tariq Rashid Architecture)
    train_csv: str = "data/fashion/fashion-mnist_train.csv"
    test_csv: str = "data/fashion/fashion-mnist_test.csv"
    
    input_nodes: int = 784
    hidden_nodes: int = 200
    output_nodes: int = 10
    learning_rate: float = 0.1
    epochs: int = 5

    # 2. Loading Dataset
    try:
        X_train, y_train, X_test, y_test = load_fashion_mnist(train_csv, test_csv)
    except FileNotFoundError as e:
        print(e)
        return

    # 3. Initialize Custom Neural Network
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 4. Training Loop (Sequential processing as per the original logic)
    print(f"[EXECUTION] Training Custom Core Model for {epochs} epochs...")
    for e in range(epochs):
        print(f" > Processing Epoch {e+1}/{epochs}...")
        for i in range(len(X_train)):
            # Target vector initialization (0.01 for non-targets, 0.99 for target)
            targets = np.zeros(output_nodes) + 0.01
            targets[y_train[i]] = 0.99
            
            # Perform a training step
            n.train(X_train[i], targets)

    # 5. Inference & Metrics
    print("[SYSTEM] Testing Core Model & Generating Predictions...")
    predictions: list[int] = []
    
    for i in range(len(X_test)):
        outputs: np.ndarray = n.query(X_test[i])
        predictions.append(int(np.argmax(outputs)))
    
    # Calculate confusion matrix and accuracy
    cm: np.ndarray = confusion_matrix(y_test, predictions)
    accuracy: float = float(np.trace(cm) / np.sum(cm) * 100)

    print(f"\n[FINAL METRICS] Model: Custom Core (Tariq Rashid)")
    print(f"[FINAL METRICS] Accuracy: {accuracy:.2f}%")

    # 6. Topological Visualization
    fashion_labels: list[str] = [
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
        "Sandal", "Shirt", "Sneaker", "Bag", "Boot"
    ]
    
    generate_3d_topology(
        conf_matrix=cm,
        accuracy=accuracy,
        labels=fashion_labels,
        model_title="Custom Core MLP (Tariq Rashid Baseline)",
        output_filename="core_fashion_topology"
    )

if __name__ == "__main__":
    main()