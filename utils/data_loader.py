import os
import numpy as np
import pandas as pd
from typing import Tuple

def load_fashion_mnist(train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and normalizes the Fashion-MNIST dataset from CSV files.
    Applies vectorized normalization to the range [0.01, 1.0].
    """
    
    # 1. Check if files exist before processing
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"[ERROR] Dataset files not found at:\n{train_path}\n{test_path}"
        )

    print(f"[SYSTEM] Loading data from {os.path.basename(train_path)}...")
    
    # 2. High-speed CSV parsing
    # Using skiprows=1 because Fashion-MNIST CSVs usually have a header row
    train_df = pd.read_csv(train_path, skiprows=1, header=None)
    test_df = pd.read_csv(test_path, skiprows=1, header=None)
    
    # 3. Extract Labels (Column 0)
    y_train: np.ndarray = train_df.iloc[:, 0].values.astype(int)
    y_test: np.ndarray = test_df.iloc[:, 0].values.astype(int)
    
    # 4. Extract Pixels (Columns 1 to 785) and Normalize
    # We convert to float and scale: (val / 255.0 * 0.99) + 0.01
    # This keeps inputs within the active range of the sigmoid function
    print("[SYSTEM] Performing vectorized normalization [0.01, 1.0]...")
    
    X_train: np.ndarray = (train_df.iloc[:, 1:].values.astype(float) / 255.0 * 0.99) + 0.01
    X_test: np.ndarray = (test_df.iloc[:, 1:].values.astype(float) / 255.0 * 0.99) + 0.01
    
    print(f"[SUCCESS] Loaded {len(X_train)} training and {len(X_test)} test samples.")
    
    return X_train, y_train, X_test, y_test