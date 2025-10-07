# Machine Learning Module - Implementation Guide

**Status:** Production Ready
**Compatibility:** Local Python, Google Colab, Jupyter
**Skill Levels:** Beginner to Advanced
**Estimated Implementation Time:** 14-18 hours

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Complete Implementation](#2-complete-implementation)
3. [Setup Instructions](#3-setup-instructions)
4. [Google Colab Integration](#4-google-colab-integration)
5. [Configuration System](#5-configuration-system)
6. [Adding New Algorithms](#6-adding-new-algorithms)
7. [Student Activities](#7-student-activities)
8. [Testing & Validation](#8-testing--validation)

---

## 1. Architecture Overview

### 1.1 Module Structure

```
modules/machine_learning/
├── __init__.py
├── config.py                    # Configuration and hyperparameters
├── core/
│   ├── __init__.py
│   ├── dataset.py              # Dataset generation and management
│   ├── base_model.py           # Base class for ML models
│   └── metrics.py              # Evaluation metrics
├── algorithms/
│   ├── __init__.py
│   ├── linear_regression.py    # Linear regression
│   ├── logistic_regression.py  # Logistic regression
│   ├── decision_tree.py        # Decision tree
│   ├── kmeans.py               # K-means clustering
│   └── neural_network.py       # Simple feedforward NN
├── ui/
│   ├── __init__.py
│   ├── visualizer.py           # Main visualization
│   ├── data_viz.py             # Data point visualization
│   ├── decision_boundary.py    # Decision boundary plotting
│   └── learning_curve.py       # Training progress
└── main.py                      # Entry point
```

### 1.2 Learning Objectives

Students will understand:
- **Supervised Learning**: Learning from labeled examples
- **Unsupervised Learning**: Finding patterns without labels
- **Overfitting vs Underfitting**: Model complexity trade-offs
- **Regularization**: Preventing overfitting
- **Feature Engineering**: Transforming raw data
- **Gradient Descent**: Optimization fundamentals
- **Evaluation**: Metrics for model assessment

---

## 2. Complete Implementation

### 2.1 Configuration

```python
# modules/machine_learning/config.py
"""Configuration for Machine Learning Module"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class MLConfig:
    """Configuration for ML module"""

    # Display settings
    WINDOW_WIDTH: int = 1400
    WINDOW_HEIGHT: int = 800
    PLOT_WIDTH: int = 600
    PLOT_HEIGHT: int = 600
    PANEL_WIDTH: int = 400
    FPS: int = 60

    # Dataset settings
    NUM_SAMPLES: int = 100
    NOISE_LEVEL: float = 0.2
    TRAIN_TEST_SPLIT: float = 0.7  # 70% train, 30% test

    # Regression settings
    LEARNING_RATE: float = 0.01
    NUM_ITERATIONS: int = 1000
    POLYNOMIAL_DEGREE: int = 1  # 1=linear, 2=quadratic, etc.
    REGULARIZATION_LAMBDA: float = 0.0  # L2 regularization

    # Classification settings
    NUM_CLASSES: int = 2
    DECISION_BOUNDARY_RESOLUTION: int = 100

    # Clustering settings
    K_CLUSTERS: int = 3
    MAX_KMEANS_ITERATIONS: int = 100

    # Neural Network settings
    HIDDEN_LAYERS: list = None  # [64, 32] for 2 hidden layers
    ACTIVATION: str = 'relu'  # 'relu', 'sigmoid', 'tanh'
    BATCH_SIZE: int = 32
    EPOCHS: int = 100

    # Visualization settings
    SHOW_TRAIN_DATA: bool = True
    SHOW_TEST_DATA: bool = True
    SHOW_MODEL: bool = True
    SHOW_LOSS_CURVE: bool = True
    ANIMATE_TRAINING: bool = True
    UPDATE_EVERY_N_STEPS: int = 10

    # Colors
    COLOR_BACKGROUND: Tuple[int, int, int] = (30, 30, 46)
    COLOR_TRAIN_POINT: Tuple[int, int, int] = (92, 124, 250)
    COLOR_TEST_POINT: Tuple[int, int, int] = (255, 121, 198)
    COLOR_MODEL_LINE: Tuple[int, int, int] = (80, 250, 123)
    COLOR_DECISION_BOUNDARY: Tuple[int, int, int] = (139, 233, 253)
    COLOR_TEXT: Tuple[int, int, int] = (248, 248, 242)
    COLOR_UI_BG: Tuple[int, int, int] = (40, 42, 54)
    COLOR_GRID: Tuple[int, int, int] = (68, 71, 90)

    # Class colors for classification
    COLOR_CLASS_0: Tuple[int, int, int] = (255, 121, 198)  # Pink
    COLOR_CLASS_1: Tuple[int, int, int] = (139, 233, 253)  # Cyan
    COLOR_CLASS_2: Tuple[int, int, int] = (80, 250, 123)   # Green
    COLOR_CLASS_3: Tuple[int, int, int] = (241, 250, 140)  # Yellow

    def __post_init__(self):
        if self.HIDDEN_LAYERS is None:
            self.HIDDEN_LAYERS = [64, 32]


# Global config
config = MLConfig()


# Presets
PRESETS = {
    'default': MLConfig(),

    'simple_regression': MLConfig(
        NUM_SAMPLES=50,
        POLYNOMIAL_DEGREE=1,
        LEARNING_RATE=0.05,
        NUM_ITERATIONS=500,
    ),

    'overfitting_demo': MLConfig(
        NUM_SAMPLES=30,
        POLYNOMIAL_DEGREE=8,  # High degree = overfitting risk
        NOISE_LEVEL=0.3,
        REGULARIZATION_LAMBDA=0.0,
    ),

    'classification': MLConfig(
        NUM_CLASSES=3,
        NUM_SAMPLES=150,
        LEARNING_RATE=0.1,
    ),

    'clustering': MLConfig(
        K_CLUSTERS=4,
        NUM_SAMPLES=200,
    ),
}


def load_preset(name: str):
    """Load preset configuration"""
    global config
    if name in PRESETS:
        preset_config = PRESETS[name]
        for attr in dir(preset_config):
            if not attr.startswith('_') and attr.isupper():
                setattr(config, attr, getattr(preset_config, attr))
        print(f"Loaded preset: {name}")
    else:
        print(f"Unknown preset: {name}")
```

### 2.2 Dataset Generation

```python
# modules/machine_learning/core/dataset.py
"""Dataset generation and management"""

import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split

class DatasetGenerator:
    """Generate synthetic datasets for ML"""

    @staticmethod
    def linear_regression(
        n_samples: int = 100,
        noise: float = 0.2,
        slope: float = 2.0,
        intercept: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate linear regression dataset

        Returns:
            X: Features (n_samples, 1)
            y: Targets (n_samples,)
        """
        X = np.random.uniform(-5, 5, (n_samples, 1))
        y = slope * X.squeeze() + intercept + np.random.normal(0, noise, n_samples)
        return X, y

    @staticmethod
    def polynomial_regression(
        n_samples: int = 100,
        noise: float = 0.2,
        degree: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate polynomial regression dataset"""
        X = np.random.uniform(-3, 3, (n_samples, 1))
        x = X.squeeze()

        # Generate polynomial: y = x^3 - 2x^2 + x + noise
        y = x**3 - 2*x**2 + x + np.random.normal(0, noise, n_samples)

        return X, y

    @staticmethod
    def binary_classification(
        n_samples: int = 100,
        n_features: int = 2,
        separation: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate binary classification dataset (two clusters)

        Returns:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,) - 0 or 1
        """
        # Class 0
        n_class0 = n_samples // 2
        X0 = np.random.randn(n_class0, n_features) - separation
        y0 = np.zeros(n_class0)

        # Class 1
        n_class1 = n_samples - n_class0
        X1 = np.random.randn(n_class1, n_features) + separation
        y1 = np.ones(n_class1)

        # Combine
        X = np.vstack([X0, X1])
        y = np.concatenate([y0, y1])

        # Shuffle
        indices = np.random.permutation(n_samples)
        return X[indices], y[indices]

    @staticmethod
    def multiclass_classification(
        n_samples: int = 150,
        n_classes: int = 3,
        n_features: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate multi-class classification dataset"""
        samples_per_class = n_samples // n_classes
        X_list = []
        y_list = []

        for i in range(n_classes):
            # Create cluster for each class
            angle = (2 * np.pi * i) / n_classes
            center = 3 * np.array([np.cos(angle), np.sin(angle)])

            X_class = np.random.randn(samples_per_class, n_features) + center
            y_class = np.full(samples_per_class, i)

            X_list.append(X_class)
            y_list.append(y_class)

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        indices = np.random.permutation(len(y))
        return X[indices], y[indices]

    @staticmethod
    def clustering_data(
        n_samples: int = 200,
        n_clusters: int = 3,
        cluster_std: float = 1.0
    ) -> np.ndarray:
        """Generate clustering dataset"""
        X_list = []

        for i in range(n_clusters):
            # Random cluster center
            center = np.random.uniform(-10, 10, 2)

            # Generate points around center
            samples = n_samples // n_clusters
            X_cluster = np.random.randn(samples, 2) * cluster_std + center

            X_list.append(X_cluster)

        return np.vstack(X_list)


class Dataset:
    """Dataset wrapper with train/test split"""

    def __init__(self, X: np.ndarray, y: np.ndarray = None, test_size: float = 0.3):
        self.X = X
        self.y = y

        if y is not None:
            # Supervised learning - split train/test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        else:
            # Unsupervised learning - no labels
            self.X_train = X
            self.X_test = None
            self.y_train = None
            self.y_test = None

    def get_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data"""
        return self.X_train, self.y_train

    def get_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data"""
        return self.X_test, self.y_test

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all data"""
        return self.X, self.y
```

### 2.3 Linear Regression

```python
# modules/machine_learning/algorithms/linear_regression.py
"""Linear and Polynomial Regression"""

import numpy as np
from typing import Tuple

class LinearRegression:
    """Linear regression with gradient descent"""

    def __init__(
        self,
        learning_rate: float = 0.01,
        num_iterations: int = 1000,
        regularization: float = 0.0
    ):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization  # L2 regularization

        # Model parameters
        self.weights = None
        self.bias = None

        # Training history
        self.loss_history = []
        self.weight_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train model using gradient descent

        Args:
            X: Features (n_samples, n_features)
            y: Targets (n_samples,)

        Yields:
            dict: Training state for visualization
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for iteration in range(self.num_iterations):
            # Forward pass
            y_pred = self.predict(X)

            # Compute loss (MSE + regularization)
            mse = np.mean((y_pred - y) ** 2)
            reg_term = self.regularization * np.sum(self.weights ** 2)
            loss = mse + reg_term

            # Compute gradients
            error = y_pred - y
            dw = (2 / n_samples) * X.T @ error + 2 * self.regularization * self.weights
            db = (2 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Record history
            self.loss_history.append(loss)
            self.weight_history.append(self.weights.copy())

            # Yield state for visualization
            if iteration % 10 == 0 or iteration == self.num_iterations - 1:
                yield {
                    'iteration': iteration,
                    'loss': loss,
                    'weights': self.weights.copy(),
                    'bias': self.bias,
                    'predictions': y_pred,
                }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return X @ self.weights + self.bias

    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Calculate evaluation metrics"""
        y_pred = self.predict(X)

        mse = np.mean((y_pred - y) ** 2)
        rmse = np.sqrt(mse)

        # R-squared
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
        }


class PolynomialRegression(LinearRegression):
    """Polynomial regression using feature transformation"""

    def __init__(self, degree: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transform features to polynomial"""
        if X.shape[1] != 1:
            raise ValueError("Polynomial regression requires 1D input")

        x = X.squeeze()
        X_poly = np.column_stack([x**i for i in range(1, self.degree + 1)])
        return X_poly

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit polynomial model"""
        X_poly = self._transform_features(X)
        return super().fit(X_poly, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with polynomial features"""
        X_poly = self._transform_features(X)
        return super().predict(X_poly)
```

### 2.4 Logistic Regression

```python
# modules/machine_learning/algorithms/logistic_regression.py
"""Logistic Regression for binary classification"""

import numpy as np

class LogisticRegression:
    """Binary logistic regression with gradient descent"""

    def __init__(
        self,
        learning_rate: float = 0.1,
        num_iterations: int = 1000,
        regularization: float = 0.0
    ):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization

        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train logistic regression"""
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for iteration in range(self.num_iterations):
            # Forward pass
            linear = X @ self.weights + self.bias
            y_pred = self.sigmoid(linear)

            # Compute loss (binary cross-entropy)
            epsilon = 1e-7  # Prevent log(0)
            loss = -np.mean(
                y * np.log(y_pred + epsilon) +
                (1 - y) * np.log(1 - y_pred + epsilon)
            )
            loss += self.regularization * np.sum(self.weights ** 2)

            # Gradients
            error = y_pred - y
            dw = (1 / n_samples) * X.T @ error + 2 * self.regularization * self.weights
            db = (1 / n_samples) * np.sum(error)

            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.loss_history.append(loss)

            # Yield for visualization
            if iteration % 10 == 0:
                yield {
                    'iteration': iteration,
                    'loss': loss,
                    'weights': self.weights.copy(),
                    'bias': self.bias,
                    'predictions': y_pred,
                }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        linear = X @ self.weights + self.bias
        return self.sigmoid(linear)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Calculate metrics"""
        y_pred = self.predict(X)

        accuracy = np.mean(y_pred == y)

        # Confusion matrix
        tp = np.sum((y_pred == 1) & (y == 1))
        tn = np.sum((y_pred == 0) & (y == 0))
        fp = np.sum((y_pred == 1) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
```

### 2.5 K-Means Clustering

```python
# modules/machine_learning/algorithms/kmeans.py
"""K-Means clustering algorithm"""

import numpy as np
from typing import Tuple

class KMeans:
    """K-Means clustering"""

    def __init__(self, k: int = 3, max_iterations: int = 100):
        self.k = k
        self.max_iterations = max_iterations

        self.centroids = None
        self.labels = None
        self.inertia_history = []

    def fit(self, X: np.ndarray):
        """
        Fit K-Means clustering

        Args:
            X: Data points (n_samples, n_features)

        Yields:
            dict: Current clustering state
        """
        n_samples, n_features = X.shape

        # Initialize centroids randomly
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[indices].copy()

        for iteration in range(self.max_iterations):
            # Assign points to nearest centroid
            old_labels = self.labels
            self.labels = self._assign_labels(X)

            # Update centroids
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = cluster_points.mean(axis=0)

            # Calculate inertia (within-cluster sum of squares)
            inertia = self._calculate_inertia(X)
            self.inertia_history.append(inertia)

            # Yield for visualization
            yield {
                'iteration': iteration,
                'centroids': self.centroids.copy(),
                'labels': self.labels.copy(),
                'inertia': inertia,
            }

            # Check convergence
            if old_labels is not None and np.array_equal(self.labels, old_labels):
                print(f"Converged after {iteration + 1} iterations")
                break

    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid"""
        distances = np.zeros((len(X), self.k))

        for i in range(self.k):
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)

        return np.argmin(distances, axis=1)

    def _calculate_inertia(self, X: np.ndarray) -> float:
        """Calculate within-cluster sum of squares"""
        inertia = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i]) ** 2)
        return inertia

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new points"""
        return self._assign_labels(X)
```

### 2.6 Simple Neural Network

```python
# modules/machine_learning/algorithms/neural_network.py
"""Simple feedforward neural network"""

import numpy as np
from typing import List, Tuple

class NeuralNetwork:
    """Simple feedforward neural network"""

    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.01,
        activation: str = 'relu'
    ):
        """
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            learning_rate: Learning rate for gradient descent
            activation: Activation function ('relu', 'sigmoid', 'tanh')
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_name = activation

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])

            self.weights.append(w)
            self.biases.append(b)

        # Training history
        self.loss_history = []

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation_name == 'relu':
            return np.maximum(0, x)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation_name == 'tanh':
            return np.tanh(x)
        return x

    def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of activation function"""
        if self.activation_name == 'relu':
            return (x > 0).astype(float)
        elif self.activation_name == 'sigmoid':
            sig = self._activation(x)
            return sig * (1 - sig)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(x) ** 2
        return np.ones_like(x)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass

        Returns:
            output, activations, z_values
        """
        activations = [X]
        z_values = []

        current = X

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            z_values.append(z)

            # Apply activation (except last layer)
            if i < len(self.weights) - 1:
                current = self._activation(z)
            else:
                current = z  # Linear output for regression

            activations.append(current)

        return current, activations, z_values

    def backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List[np.ndarray],
        z_values: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward pass - compute gradients"""
        n_samples = X.shape[0]

        # Gradients
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        # Output layer gradient
        delta = (activations[-1] - y.reshape(-1, 1)) / n_samples

        # Backpropagate
        for i in reversed(range(len(self.weights))):
            # Gradient for weights and bias
            dw[i] = activations[i].T @ delta
            db[i] = np.sum(delta, axis=0)

            # Propagate to previous layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._activation_derivative(z_values[i-1])

        return dw, db

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Single training step"""
        # Forward pass
        output, activations, z_values = self.forward(X)

        # Compute loss
        loss = np.mean((output.squeeze() - y) ** 2)

        # Backward pass
        dw, db = self.backward(X, y, activations, z_values)

        # Update parameters
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dw[i]
            self.biases[i] -= self.learning_rate * db[i]

        return loss

    def fit_batch(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train with mini-batches"""
        for epoch in range(epochs):
            loss = self.train_step(X, y)
            self.loss_history.append(loss)

            if epoch % 10 == 0:
                yield {
                    'epoch': epoch,
                    'loss': loss,
                    'weights': [w.copy() for w in self.weights],
                }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        output, _, _ = self.forward(X)
        return output.squeeze()
```

### 2.7 Interactive Visualization

```python
# modules/machine_learning/ui/visualizer.py
"""ML Visualization using Pygame and Matplotlib"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from modules.machine_learning.config import config

class MLVisualizer:
    """Interactive ML visualizer"""

    def __init__(self, dataset_type: str = 'regression'):
        pygame.init()

        self.dataset_type = dataset_type

        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        pygame.display.set_caption(f"Machine Learning - {dataset_type.title()}")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Matplotlib figure for plots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 8))
        self.canvas = FigureCanvasAgg(self.fig)

    def render_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model,
        iteration: int = 0
    ):
        """Render regression visualization"""
        self.screen.fill(config.COLOR_BACKGROUND)

        # Update plots
        self._plot_regression(X_train, y_train, X_test, y_test, model)

        # Convert matplotlib to pygame surface
        self.canvas.draw()
        raw_data = self.canvas.get_renderer().tostring_rgb()
        size = self.canvas.get_width_height()
        plot_surface = pygame.image.fromstring(raw_data, size, "RGB")

        # Display plot
        self.screen.blit(plot_surface, (50, 50))

        # Display metrics
        self._render_metrics_panel(model, X_train, y_train, X_test, y_test, iteration)

        pygame.display.flip()
        self.clock.tick(config.FPS)

    def _plot_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model
    ):
        """Plot regression data and model"""
        self.axes[0, 0].clear()
        self.axes[0, 1].clear()
        self.axes[1, 0].clear()
        self.axes[1, 1].clear()

        # Plot 1: Data and model fit
        ax = self.axes[0, 0]

        # Plot training data
        ax.scatter(X_train, y_train, c='blue', alpha=0.6, label='Train', s=50)

        # Plot test data
        if X_test is not None and len(X_test) > 0:
            ax.scatter(X_test, y_test, c='red', alpha=0.6, label='Test', s=50)

        # Plot model predictions
        if model.weights is not None:
            x_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
            y_line = model.predict(x_line)
            ax.plot(x_line, y_line, 'g-', linewidth=2, label='Model')

        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title('Data and Model Fit')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Loss curve
        ax = self.axes[0, 1]
        if model.loss_history:
            ax.plot(model.loss_history, 'b-', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss (MSE)')
            ax.set_title('Training Loss')
            ax.grid(True, alpha=0.3)

        # Plot 3: Residuals
        ax = self.axes[1, 0]
        if model.weights is not None:
            y_pred_train = model.predict(X_train)
            residuals = y_train - y_pred_train

            ax.scatter(y_pred_train, residuals, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot')
            ax.grid(True, alpha=0.3)

        # Plot 4: Metrics
        ax = self.axes[1, 1]
        ax.axis('off')

        if model.weights is not None:
            train_metrics = model.get_metrics(X_train, y_train)
            test_metrics = model.get_metrics(X_test, y_test) if X_test is not None else {}

            metrics_text = f"""
            Training Metrics:
              MSE:  {train_metrics.get('mse', 0):.4f}
              RMSE: {train_metrics.get('rmse', 0):.4f}
              R²:   {train_metrics.get('r2', 0):.4f}

            Test Metrics:
              MSE:  {test_metrics.get('mse', 0):.4f}
              RMSE: {test_metrics.get('rmse', 0):.4f}
              R²:   {test_metrics.get('r2', 0):.4f}
            """

            ax.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                   verticalalignment='center')

        self.fig.tight_layout()

    def _render_metrics_panel(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        iteration
    ):
        """Render metrics panel on pygame surface"""
        panel_x = config.PLOT_WIDTH + 100
        y_offset = 50

        # Title
        title = self.font.render("Training Progress", True, config.COLOR_TEXT)
        self.screen.blit(title, (panel_x, y_offset))
        y_offset += 40

        # Iteration
        iter_text = self.small_font.render(
            f"Iteration: {iteration}",
            True,
            config.COLOR_TEXT
        )
        self.screen.blit(iter_text, (panel_x, y_offset))
        y_offset += 30

        # Loss
        if model.loss_history:
            loss_text = self.small_font.render(
                f"Loss: {model.loss_history[-1]:.4f}",
                True,
                config.COLOR_TEXT
            )
            self.screen.blit(loss_text, (panel_x, y_offset))
```

### 2.8 Main Application for Regression

```python
# modules/machine_learning/main.py
"""Main application for ML module"""

import pygame
import sys
from modules.machine_learning.config import config
from modules.machine_learning.core.dataset import DatasetGenerator, Dataset
from modules.machine_learning.algorithms.linear_regression import LinearRegression, PolynomialRegression
from modules.machine_learning.ui.visualizer import MLVisualizer

class MLApp:
    """Machine Learning application"""

    def __init__(self, task: str = 'regression'):
        self.task = task

        # Generate dataset
        if task == 'regression':
            X, y = DatasetGenerator.linear_regression(
                n_samples=config.NUM_SAMPLES,
                noise=config.NOISE_LEVEL
            )
        elif task == 'polynomial':
            X, y = DatasetGenerator.polynomial_regression(
                n_samples=config.NUM_SAMPLES,
                noise=config.NOISE_LEVEL
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        self.dataset = Dataset(X, y, test_size=1-config.TRAIN_TEST_SPLIT)

        # Create model
        if task == 'polynomial':
            self.model = PolynomialRegression(
                degree=config.POLYNOMIAL_DEGREE,
                learning_rate=config.LEARNING_RATE,
                num_iterations=config.NUM_ITERATIONS,
                regularization=config.REGULARIZATION_LAMBDA
            )
        else:
            self.model = LinearRegression(
                learning_rate=config.LEARNING_RATE,
                num_iterations=config.NUM_ITERATIONS,
                regularization=config.REGULARIZATION_LAMBDA
            )

        # Visualizer
        self.visualizer = MLVisualizer(task)

        # Control
        self.running = True
        self.training = False
        self.train_generator = None

        print("Machine Learning Module - Regression")
        print("=" * 50)
        print(f"Dataset: {config.NUM_SAMPLES} samples")
        print(f"Train/Test split: {config.TRAIN_TEST_SPLIT:.0%}/{1-config.TRAIN_TEST_SPLIT:.0%}")
        print("\nControls:")
        print("  SPACE: Start/Pause training")
        print("  R: Reset model")
        print("  N: New dataset")
        print("  Q: Quit")

    def start_training(self):
        """Start training"""
        X_train, y_train = self.dataset.get_train()
        self.train_generator = self.model.fit(X_train, y_train)
        self.training = True

    def step_training(self):
        """Execute one training step"""
        if self.train_generator:
            try:
                state = next(self.train_generator)
                # Update visualization with current state
                return state
            except StopIteration:
                self.training = False
                print("\nTraining complete!")
                X_test, y_test = self.dataset.get_test()
                test_metrics = self.model.get_metrics(X_test, y_test)
                print(f"Test RMSE: {test_metrics['rmse']:.4f}")
                print(f"Test R²: {test_metrics['r2']:.4f}")

    def handle_events(self):
        """Handle events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not self.training and not self.train_generator:
                        self.start_training()
                    else:
                        self.training = not self.training

                elif event.key == pygame.K_r:
                    self.reset_model()

                elif event.key == pygame.K_n:
                    self.new_dataset()

                elif event.key == pygame.K_q:
                    self.running = False

    def reset_model(self):
        """Reset model"""
        self.model = LinearRegression(
            learning_rate=config.LEARNING_RATE,
            num_iterations=config.NUM_ITERATIONS,
            regularization=config.REGULARIZATION_LAMBDA
        )
        self.train_generator = None
        self.training = False
        print("Model reset")

    def new_dataset(self):
        """Generate new dataset"""
        X, y = DatasetGenerator.linear_regression(
            n_samples=config.NUM_SAMPLES,
            noise=config.NOISE_LEVEL
        )
        self.dataset = Dataset(X, y, test_size=1-config.TRAIN_TEST_SPLIT)
        self.reset_model()
        print("New dataset generated")

    def update(self):
        """Update application"""
        if self.training:
            self.step_training()

    def render(self):
        """Render application"""
        X_train, y_train = self.dataset.get_train()
        X_test, y_test = self.dataset.get_test()

        self.visualizer.render_regression(
            X_train, y_train,
            X_test, y_test,
            self.model,
            iteration=len(self.model.loss_history)
        )

    def run(self):
        """Main loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.render()

        pygame.quit()
        sys.exit()


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='ML Module')
    parser.add_argument('--task', type=str, default='regression',
                       choices=['regression', 'polynomial', 'classification', 'clustering'])
    args = parser.parse_args()

    app = MLApp(task=args.task)
    app.run()


if __name__ == '__main__':
    main()
```

---

## 3. Setup Instructions

```bash
# Install dependencies
pip install numpy matplotlib scikit-learn pygame

# Run module
python -m modules.machine_learning.main

# Or with task
python -m modules.machine_learning.main --task polynomial
```

---

## 4. Google Colab Integration

```python
# modules/machine_learning/colab_main.py
"""Colab-compatible ML visualization"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from modules.machine_learning.core.dataset import DatasetGenerator, Dataset
from modules.machine_learning.algorithms.linear_regression import LinearRegression

def train_regression_colab(
    degree: int = 1,
    learning_rate: float = 0.01,
    iterations: int = 1000,
    n_samples: int = 100
):
    """Train regression in Colab with live visualization"""

    # Generate data
    X, y = DatasetGenerator.linear_regression(n_samples=n_samples)
    dataset = Dataset(X, y)

    # Create model
    if degree > 1:
        from modules.machine_learning.algorithms.linear_regression import PolynomialRegression
        model = PolynomialRegression(degree=degree, learning_rate=learning_rate)
    else:
        model = LinearRegression(learning_rate=learning_rate, num_iterations=iterations)

    # Train with visualization
    X_train, y_train = dataset.get_train()
    X_test, y_test = dataset.get_test()

    for state in model.fit(X_train, y_train):
        if state['iteration'] % 50 == 0:
            clear_output(wait=True)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Data and model
            axes[0].scatter(X_train, y_train, alpha=0.6, label='Train')
            axes[0].scatter(X_test, y_test, alpha=0.6, label='Test')

            x_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
            y_line = model.predict(x_line)
            axes[0].plot(x_line, y_line, 'r-', linewidth=2, label='Model')

            axes[0].set_title(f'Iteration {state["iteration"]}')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Loss curve
            axes[1].plot(model.loss_history)
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Training Loss')
            axes[1].grid(True, alpha=0.3)

            plt.show()

    return model, dataset


# Usage in Colab:
"""
!pip install -q numpy matplotlib scikit-learn

from modules.machine_learning.colab_main import train_regression_colab

# Linear regression
model, dataset = train_regression_colab(degree=1, learning_rate=0.05)

# Polynomial regression
model, dataset = train_regression_colab(degree=3, learning_rate=0.01)
"""
```

---

## 5. Student Activities

### 5.1 Beginner: Understand Overfitting

```python
from modules.machine_learning.config import config, load_preset

# Load overfitting demo preset
load_preset('overfitting_demo')

# Try different polynomial degrees
for degree in [1, 2, 5, 10]:
    config.POLYNOMIAL_DEGREE = degree
    # Run and observe: higher degree = overfitting!
```

### 5.2 Intermediate: Regularization

```python
# Test different regularization strengths
for reg_lambda in [0.0, 0.01, 0.1, 1.0]:
    config.REGULARIZATION_LAMBDA = reg_lambda
    # Run and compare train vs test error
```

### 5.3 Advanced: Implement Ridge Regression

```python
class RidgeRegression(LinearRegression):
    """Ridge regression with L2 regularization"""

    def fit(self, X, y):
        # Closed-form solution: w = (X^T X + λI)^-1 X^T y
        n_features = X.shape[1]
        I = np.eye(n_features)

        # Add regularization to normal equation
        self.weights = np.linalg.solve(
            X.T @ X + self.regularization * I,
            X.T @ y
        )

        self.bias = 0  # Assuming centered data
```

---

## Summary

This ML module provides:

✅ **4 ML Algorithms**: Linear Regression, Logistic Regression, K-Means, Neural Network
✅ **Interactive Visualization**: Real-time training progress
✅ **Dataset Generation**: Synthetic datasets with configurable noise
✅ **Comprehensive Metrics**: MSE, RMSE, R², Accuracy, Precision, Recall
✅ **Student Activities**: Beginner → Advanced
✅ **Colab Support**: Matplotlib-based visualization
✅ **Extensible**: Easy to add new algorithms

**Learning Outcomes:**
- Understand supervised vs unsupervised learning
- Recognize overfitting and how to prevent it
- Experiment with hyperparameters
- Visualize model learning process
- Implement custom ML algorithms

**Perfect for teaching:**
- Introduction to Machine Learning
- Data Science fundamentals
- Model evaluation and selection
- Optimization basics
