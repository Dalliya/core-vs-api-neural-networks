import sys
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Adjust system path to allow imports from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.sklearn_pipeline import SklearnMLPPipeline
from utils.data_loader import load_fashion_mnist
from visualizations.plotter import generate_3d_topology

def main() -> None:
    # 1. Configuration of paths
    train_csv = "data/fashion/fashion-mnist_train.csv"
    test_csv = "data/fashion/fashion-mnist_test.csv"
    
    # 2. Loading Dataset
    try:
        X_train_full, y_train_full, X_test, y_test = load_fashion_mnist(train_csv, test_csv)
    except FileNotFoundError as e:
        print(e)
        return

    # Split for Optuna validation (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # 3. Initialize and Run API Pipeline
    # n_trials=10 for a quick test, increase to 20+ for better results
    pipeline = SklearnMLPPipeline(n_trials=10) 
    pipeline.optimize_and_train(X_train, y_train, X_val, y_val)

    # 4. Inference & Metrics
    print("[SYSTEM] Generating predictions for test set...")
    predictions = pipeline.predict(X_test)

    # Calculate Confusion Matrix (Requirement)
    cm = confusion_matrix(y_test, predictions)
    accuracy = float(np.trace(cm) / np.sum(cm) * 100)
    
    best_act = pipeline.best_params['activation']
    print(f"\n[FINAL METRICS] Model: scikit-learn MLP")
    print(f"[FINAL METRICS] Best Activation: {best_act}")
    print(f"[FINAL METRICS] Accuracy: {accuracy:.2f}%")

    # 5. Visualizing results using your custom 3D logic
    fashion_labels = [
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
        "Sandal", "Shirt", "Sneaker", "Bag", "Boot"
    ]
    
    generate_3d_topology(
        conf_matrix=cm,
        accuracy=accuracy,
        labels=fashion_labels,
        model_title=f"API MLP (Activation: {best_act.upper()})",
        output_filename=f"api_fashion_topology"
    )

if __name__ == "__main__":
    main()