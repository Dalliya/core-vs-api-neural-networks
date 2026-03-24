import numpy as np
import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from typing import Dict, Any, Optional

optuna.logging.set_verbosity(optuna.logging.WARNING)

class SklearnMLPPipeline:
    """
    Wrapper for scikit-learn MLPClassifier utilizing Optuna for hyperparameter optimization.
    Implements specific requirements: evaluating 'identity', 'logistic', 'tanh', and 'relu' activations.
    """

    def __init__(self, n_trials: int = 20) -> None:
        self.n_trials = n_trials
        self.best_params: Dict[str, Any] = {}
        self.best_model: Optional[MLPClassifier] = None

    def _objective(
        self, 
        trial: optuna.Trial, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> float:
        """
        Optuna objective function.
        """
        activation = trial.suggest_categorical(
            'activation', 
            ['identity', 'logistic', 'tanh', 'relu']
        )
        hidden_layer_size = trial.suggest_int('hidden_layer_size', 50, 200)
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True)

        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_layer_size,),
            activation=activation,
            solver='adam',
            learning_rate_init=learning_rate_init,
            max_iter=50, 
            random_state=42,
            early_stopping=True
        )
        
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_val)
        
        return float(accuracy_score(y_val, predictions))

    def optimize_and_train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> None:
        """
        Executes Optuna trial runs and trains the final model using optimal parameters.
        """
        print(f"[SYSTEM] Starting Optuna optimization ({self.n_trials} trials)...")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val), 
            n_trials=self.n_trials
        )
        
        self.best_params = study.best_params
        print(f"[SYSTEM] Optimal parameters found: {self.best_params}")
        
        print("[SYSTEM] Training final scikit-learn model with optimal parameters...")
        self.best_model = MLPClassifier(
            hidden_layer_sizes=(self.best_params['hidden_layer_size'],),
            activation=self.best_params['activation'],
            solver='adam',
            learning_rate_init=self.best_params['learning_rate_init'],
            max_iter=300, 
            random_state=42
        )
        
        X_full = np.vstack((X_train, X_val))
        y_full = np.concatenate((y_train, y_val))
        
        self.best_model.fit(X_full, y_full)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generates predictions.
        """
        if self.best_model is None:
            raise ValueError("Model is not trained. Call optimize_and_train() first.")
        return self.best_model.predict(X_test)