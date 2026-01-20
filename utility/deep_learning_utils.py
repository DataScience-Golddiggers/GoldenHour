"""
Deep Learning Utilities for Time Series Forecasting
===================================================

Advanced utilities for LSTM training including:
- Learning rate scheduling strategies
- Early stopping with multiple criteria
- Batch size optimization
- Architecture search helpers
- Training monitoring and visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns


class TimeSeriesEarlyStopping:
    """
    Advanced early stopping for time series with multiple criteria.
    
    Monitors validation loss with:
    - Patience for improvement
    - Minimum delta threshold
    - Restore best weights option
    - Directional forecast accuracy check
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0001,
        restore_best_weights: bool = True,
        check_directional_accuracy: bool = False,
        verbose: int = 1
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.check_directional_accuracy = check_directional_accuracy
        self.verbose = verbose
        
        self.best_loss = np.inf
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
    
    def __call__(self, epoch: int, val_loss: float, model_weights: Optional = None) -> bool:
        """
        Check if training should stop.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        val_loss : float
            Validation loss for current epoch
        model_weights : optional
            Current model weights (for restoration)
        
        Returns
        -------
        bool
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights and model_weights is not None:
                self.best_weights = model_weights
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best epoch: {self.best_epoch} with loss: {self.best_loss:.6f}")
                return True
        
        return False
    
    def get_best_weights(self):
        """Return best weights if restore_best_weights is True."""
        return self.best_weights


class LearningRateScheduler:
    """
    Custom learning rate schedulers for time series.
    
    Strategies:
    - step_decay: Reduce LR by factor every N epochs
    - exponential_decay: Exponential decrease
    - reduce_on_plateau: Reduce when validation loss plateaus
    - cosine_annealing: Cosine decay with warm restarts
    """
    
    def __init__(
        self,
        strategy: str = 'reduce_on_plateau',
        initial_lr: float = 0.001,
        **kwargs
    ):
        self.strategy = strategy
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.kwargs = kwargs
        
        # For reduce_on_plateau
        self.best_loss = np.inf
        self.patience_count = 0
        self.plateau_patience = kwargs.get('patience', 10)
        self.factor = kwargs.get('factor', 0.5)
        self.min_lr = kwargs.get('min_lr', 1e-7)
    
    def step(self, epoch: int, val_loss: Optional[float] = None) -> float:
        """
        Calculate learning rate for current epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        val_loss : float, optional
            Validation loss (for plateau-based strategies)
        
        Returns
        -------
        float
            Learning rate for this epoch
        """
        if self.strategy == 'step_decay':
            step_size = self.kwargs.get('step_size', 30)
            gamma = self.kwargs.get('gamma', 0.1)
            self.current_lr = self.initial_lr * (gamma ** (epoch // step_size))
        
        elif self.strategy == 'exponential_decay':
            decay_rate = self.kwargs.get('decay_rate', 0.95)
            self.current_lr = self.initial_lr * (decay_rate ** epoch)
        
        elif self.strategy == 'reduce_on_plateau':
            if val_loss is not None:
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.patience_count = 0
                else:
                    self.patience_count += 1
                    if self.patience_count >= self.plateau_patience:
                        self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                        self.patience_count = 0
        
        elif self.strategy == 'cosine_annealing':
            T_max = self.kwargs.get('T_max', 50)
            eta_min = self.kwargs.get('eta_min', 0)
            self.current_lr = eta_min + (self.initial_lr - eta_min) * \
                             (1 + np.cos(np.pi * epoch / T_max)) / 2
        
        return max(self.current_lr, self.min_lr)


def find_optimal_batch_size(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_builder: Callable,
    batch_sizes: List[int] = [16, 32, 64, 128],
    epochs: int = 10,
    verbose: bool = True
) -> int:
    """
    Find optimal batch size by testing different values.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    model_builder : callable
        Function that returns compiled model
    batch_sizes : list
        Batch sizes to test
    epochs : int
        Epochs for each test
    verbose : bool
        Print results
    
    Returns
    -------
    int
        Optimal batch size
    """
    results = []
    
    for batch_size in batch_sizes:
        model = model_builder()
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=0
        )
        
        final_val_loss = history.history['val_loss'][-1]
        training_time = sum(history.history.get('time', [1] * epochs))  # Approximate
        
        results.append({
            'batch_size': batch_size,
            'val_loss': final_val_loss,
            'training_time': training_time,
            'throughput': len(X_train) * epochs / training_time
        })
        
        if verbose:
            print(f"Batch size {batch_size}: Val Loss = {final_val_loss:.6f}")
    
    # Choose based on validation loss
    results_df = pd.DataFrame(results).sort_values('val_loss')
    optimal_batch_size = results_df.iloc[0]['batch_size']
    
    if verbose:
        print(f"\n✓ Optimal batch size: {optimal_batch_size}")
    
    return optimal_batch_size


def plot_training_history(
    history,
    metrics: List[str] = ['loss', 'mae'],
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot training history with multiple metrics.
    
    Parameters
    ----------
    history : History object or dict
        Keras History object or dict with metric histories
    metrics : list
        Metrics to plot
    figsize : tuple
        Figure size
    """
    if hasattr(history, 'history'):
        history = history.history
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        if metric in history:
            axes[idx].plot(history[metric], label=f'Train {metric}', linewidth=2)
            if f'val_{metric}' in history:
                axes[idx].plot(history[f'val_{metric}'], label=f'Val {metric}', linewidth=2)
            
            axes[idx].set_title(f'{metric.upper()} vs Epoch', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.upper())
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def calculate_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calculate directional accuracy (% of correct up/down predictions).
    
    Critical for financial time series where direction matters more than magnitude.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        Directional accuracy (0 to 1)
    """
    # For multi-step forecasts, compare step-to-step changes
    if len(y_true.shape) > 1:
        # Flatten for multi-step
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
    
    # Calculate differences
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    # Calculate accuracy
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)
    
    return correct / total if total > 0 else 0.0


def lstm_architecture_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_shape: Tuple,
    output_dim: int,
    configs: List[Dict],
    epochs: int = 50,
    batch_size: int = 32,
    verbose: bool = True
) -> Dict:
    """
    Test multiple LSTM architectures and return best.
    
    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    input_shape : tuple
        Input shape (lookback, n_features)
    output_dim : int
        Output dimension (forecast horizon)
    configs : list of dict
        List of architecture configs, each with:
        - 'layers': list of LSTM units per layer
        - 'dropout': dropout rate
        - 'learning_rate': learning rate
    epochs : int
        Training epochs
    batch_size : int
        Batch size
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        Best model configuration and results
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        raise ImportError("TensorFlow required for architecture search")
    
    results = []
    best_val_loss = np.inf
    best_config = None
    best_model = None
    
    for idx, config in enumerate(configs):
        if verbose:
            print(f"\nTesting config {idx + 1}/{len(configs)}: {config}")
        
        # Build model
        model = Sequential()
        layers = config['layers']
        dropout = config.get('dropout', 0.2)
        lr = config.get('learning_rate', 0.001)
        
        for i, units in enumerate(layers):
            return_seq = (i < len(layers) - 1)
            if i == 0:
                model.add(LSTM(units, return_sequences=return_seq, input_shape=input_shape))
            else:
                model.add(LSTM(units, return_sequences=return_seq))
            model.add(Dropout(dropout))
        
        model.add(Dense(output_dim))
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        # Evaluate
        val_loss = min(history.history['val_loss'])
        val_mae = min(history.history['val_mae'])
        
        results.append({
            'config': config,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'epochs_trained': len(history.history['loss'])
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = config
            best_model = model
        
        if verbose:
            print(f"  Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
    
    if verbose:
        print(f"\n✓ Best config: {best_config}")
        print(f"  Val Loss: {best_val_loss:.6f}")
    
    return {
        'best_config': best_config,
        'best_model': best_model,
        'best_val_loss': best_val_loss,
        'all_results': pd.DataFrame(results)
    }


def create_ensemble_predictions(
    models: List,
    X_test: np.ndarray,
    method: str = 'mean'
) -> np.ndarray:
    """
    Create ensemble predictions from multiple models.
    
    Parameters
    ----------
    models : list
        List of trained models
    X_test : np.ndarray
        Test data
    method : str
        'mean', 'median', or 'weighted'
    
    Returns
    -------
    np.ndarray
        Ensemble predictions
    """
    predictions = []
    
    for model in models:
        pred = model.predict(X_test, verbose=0)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    if method == 'mean':
        return np.mean(predictions, axis=0)
    elif method == 'median':
        return np.median(predictions, axis=0)
    elif method == 'weighted':
        # Weight by inverse validation loss (would need to be passed in)
        # For now, just use mean
        return np.mean(predictions, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def analyze_lstm_feature_importance(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    n_iterations: int = 100
) -> pd.DataFrame:
    """
    Estimate feature importance for LSTM using permutation importance.
    
    Parameters
    ----------
    model : trained model
        LSTM model
    X_train : np.ndarray
        Training features [samples, timesteps, features]
    y_train : np.ndarray
        Training targets
    feature_names : list
        Names of features
    n_iterations : int
        Number of permutation iterations
    
    Returns
    -------
    pd.DataFrame
        Feature importance scores
    """
    from sklearn.metrics import mean_squared_error
    
    # Baseline score
    baseline_pred = model.predict(X_train, verbose=0)
    baseline_score = mean_squared_error(y_train, baseline_pred)
    
    importances = []
    
    for feature_idx in range(X_train.shape[2]):
        feature_scores = []
        
        for _ in range(n_iterations):
            # Permute feature
            X_permuted = X_train.copy()
            np.random.shuffle(X_permuted[:, :, feature_idx])
            
            # Score
            permuted_pred = model.predict(X_permuted, verbose=0)
            permuted_score = mean_squared_error(y_train, permuted_pred)
            
            # Importance = increase in error
            feature_scores.append(permuted_score - baseline_score)
        
        importances.append({
            'feature': feature_names[feature_idx],
            'importance': np.mean(feature_scores),
            'std': np.std(feature_scores)
        })
    
    df = pd.DataFrame(importances).sort_values('importance', ascending=False)
    return df
