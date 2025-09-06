"""
Neural Network from scratch using ONLY NumPy and Pandas
------------------------------------------------------

Este código implementa uma rede neural totalmente do zero, sem frameworks de deep learning
como TensorFlow ou PyTorch, apenas com NumPy e Pandas. O objetivo é mostrar como os
princípios básicos funcionam: camadas, ativações, propagação para frente e para trás,
funções de perda, otimizadores, etc.

No final há um exemplo de classificação em um dataset sintético.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Any, Dict

# Fixamos a semente para reprodutibilidade
np.random.seed(42)

# ------------------------- Funções utilitárias -------------------------
def to_numpy(x):
    """Converte DataFrame ou Series do Pandas para NumPy array."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.values.astype(float)
    return np.asarray(x, dtype=float)


def one_hot(y, num_classes=None):
    """Codificação one-hot para variáveis de classe."""
    y_arr = to_numpy(y).astype(int).reshape(-1)
    n = y_arr.size
    k = int(num_classes if num_classes is not None else (y_arr.max() + 1))
    out = np.zeros((n, k), dtype=float)
    out[np.arange(n), y_arr] = 1.0
    return out


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42):
    """Divide os dados em treino e teste."""
    X = to_numpy(X)
    y = to_numpy(y)
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ------------------------- Camadas -------------------------
class Dense:
    """Camada totalmente conectada (W·X + b)."""
    def __init__(self, in_features, out_features, weight_scale=None, he=True):
        # Inicialização He ou Xavier para os pesos
        if weight_scale is None:
            if he:
                weight_scale = np.sqrt(2.0 / in_features)
            else:
                weight_scale = np.sqrt(1.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * weight_scale
        self.b = np.zeros((1, out_features))
        self.grads = {"W": np.zeros_like(self.W), "b": np.zeros_like(self.b)}
        self.cache_input = None

    def forward(self, X, training=False):
        """Propagação para frente: aplica W·X + b."""
        self.cache_input = X
        return X @ self.W + self.b

    def backward(self, dZ, l2=0.0):
        """Retropropagação: calcula gradientes da camada."""
        X = self.cache_input
        self.grads["W"] = X.T @ dZ + l2 * self.W
        self.grads["b"] = np.sum(dZ, axis=0, keepdims=True)
        return dZ @ self.W.T


class ReLU:
    """Função de ativação ReLU."""
    def __init__(self):
        self.cache = None

    def forward(self, X, training=False):
        self.cache = X
        return np.maximum(0, X)

    def backward(self, dA):
        return dA * (self.cache > 0)


class Sigmoid:
    """Função de ativação Sigmoid."""
    def __init__(self):
        self.cache = None

    def forward(self, X, training=False):
        A = 1.0 / (1.0 + np.exp(-X))
        self.cache = A
        return A

    def backward(self, dA):
        A = self.cache
        return dA * A * (1 - A)


class Tanh:
    """Função de ativação Tanh."""
    def __init__(self):
        self.cache = None

    def forward(self, X, training=False):
        A = np.tanh(X)
        self.cache = A
        return A

    def backward(self, dA):
        A = self.cache
        return dA * (1 - A ** 2)


class Softmax:
    """Função de ativação Softmax (para classificação multiclasse)."""
    def __init__(self):
        self.cache = None

    def forward(self, X, training=False):
        shift = X - np.max(X, axis=1, keepdims=True)  # estabilização numérica
        exps = np.exp(shift)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.cache = probs
        return probs

    def backward(self, dA):
        return dA  # já recebe (p - y) da loss

# ------------------------- Funções de perda -------------------------
class CrossEntropyLoss:
    """Perda de entropia cruzada para classificação."""
    def __init__(self):
        self.preds = None
        self.targets = None

    def forward(self, probs, y_true_onehot, eps=1e-12):
        self.preds = probs
        self.targets = y_true_onehot
        clipped = np.clip(probs, eps, 1 - eps)
        return -np.mean(np.sum(y_true_onehot * np.log(clipped), axis=1))

    def backward(self):
        N = self.targets.shape[0]
        return (self.preds - self.targets) / N


class MSELoss:
    """Erro quadrático médio (para regressão)."""
    def __init__(self):
        self.preds = None
        self.targets = None

    def forward(self, y_pred, y_true):
        self.preds = y_pred
        self.targets = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        N = self.targets.shape[0]
        return (2.0 / N) * (self.preds - self.targets)

# ------------------------- Otimizadores -------------------------
class SGD:
    """Gradiente descendente estocástico com momento."""
    def __init__(self, lr=1e-2, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.v: Dict[int, Dict[str, np.ndarray]] = {}

    def step(self, params: List[Dict[str, np.ndarray]]):
        for i, p in enumerate(params):
            if i not in self.v:
                self.v[i] = {k: np.zeros_like(v) for k, v in p.items()}
            for k in p:
                self.v[i][k] = self.momentum * self.v[i][k] + (1 - self.momentum) * p[k]
                p[k] -= self.lr * self.v[i][k]


class Adam:
    """Otimizador Adam (mais sofisticado que SGD)."""
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params: List[Dict[str, np.ndarray]]):
        self.t += 1
        for i, p in enumerate(params):
            if i not in self.m:
                self.m[i] = {k: np.zeros_like(v) for k, v in p.items()}
                self.v[i] = {k: np.zeros_like(v) for k, v in p.items()}
            for k in p:
                self.m[i][k] = self.beta1 * self.m[i][k] + (1 - self.beta1) * p[k]
                self.v[i][k] = self.beta2 * self.v[i][k] + (1 - self.beta2) * (p[k] ** 2)
                m_hat = self.m[i][k] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][k] / (1 - self.beta2 ** self.t)
                p[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ------------------------- Modelo principal -------------------------
class NeuralNetwork:
    """Classe que junta as camadas, perda e otimizador em uma rede neural completa."""
    def __init__(self, layers: List[Any], loss="cross_entropy", l2=0.0, optimizer="adam", lr=1e-3, seed=42):
        self.layers = layers
        self.l2 = l2
        self.seed = seed
        self.is_classification = (loss == "cross_entropy")
        self.loss_fn = CrossEntropyLoss() if self.is_classification else MSELoss()
        self.opt = Adam(lr=lr) if optimizer == "adam" else SGD(lr=lr, momentum=0.9)

    def _params_and_grads(self):
        """Coleta todos os gradientes dos layers para o otimizador."""
        return [{"W": l.grads["W"], "b": l.grads["b"]} for l in self.layers if isinstance(l, Dense)]

    def forward(self, X, training=False):
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, dloss):
        grad = dloss
        for layer in reversed(self.layers):
            if isinstance(layer, Dense):
                grad = layer.backward(grad, l2=self.l2)
            else:
                grad = layer.backward(grad)

    def predict(self, X):
        X = to_numpy(X)
        out = self.forward(X, training=False)
        if self.is_classification and out.ndim == 2 and out.shape[1] > 1:
            return np.argmax(out, axis=1)
        return out

    def fit(self, X, y, epochs=50, batch_size=32, verbose=True, val_split=0.2, patience=10, class_count=None):
        X = to_numpy(X)
        y = to_numpy(y)

        # Prepara os rótulos (one-hot para classificação)
        y_oh = one_hot(y, num_classes=class_count) if self.is_classification else y

        # Divide em treino e validação
        if val_split and 0 < val_split < 1:
            X_train, X_val, y_train, y_val = train_test_split(X, y_oh, test_size=val_split, random_state=self.seed)
        else:
            X_train, y_train, X_val, y_val = X, y_oh, None, None

        n = X_train.shape[0]
        best_val = np.inf
        no_improve = 0

        for epoch in range(1, epochs + 1):
            # Embaralha os dados
            idx = np.random.permutation(n)
            X_train, y_train = X_train[idx], y_train[idx]

            # Treino em mini-batches
            for start in range(0, n, batch_size):
                xb, yb = X_train[start:start+batch_size], y_train[start:start+batch_size]
                out = self.forward(xb, training=True)
                loss = self.loss_fn.forward(out, yb)
                dloss = self.loss_fn.backward()
                self.backward(dloss)
                self.opt.step(self._params_and_grads())

            # Avalia no treino e validação
            train_out = self.forward(X_train)
            train_loss = self.loss_fn.forward(train_out, y_train)
            if self.is_classification:
                train_acc = (np.argmax(train_out, axis=1) == np.argmax(y_train, axis=1)).mean()

            if X_val is not None:
                val_out = self.forward(X_val)
                val_loss = self.loss_fn.forward(val_out, y_val)
                if self.is_classification:
                    val_acc = (np.argmax(val_out, axis=1) == np.argmax(y_val, axis=1)).mean()

                if verbose:
                    print(f"Epoch {epoch:03d} | loss {train_loss:.4f} | acc {train_acc:.3f} | val_loss {val_loss:.4f} | val_acc {val_acc:.3f}")

                # Early stopping
                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping no epoch {epoch}")
                        break
            else:
                if verbose:
                    print(f"Epoch {epoch:03d} | loss {train_loss:.4f} | acc {train_acc:.3f}")

# ------------------------- Exemplo -------------------------
def make_blobs(n_samples=900, centers=3, features=2, cluster_std=1.0, random_state=42):
    """Gera dados sintéticos em formato de clusters (blobs)."""
    rng = np.random.default_rng(random_state)
    X, y = [], []
    means = rng.uniform(-6, 6, size=(centers, features))
    for k in range(centers):
        cov = np.eye(features) * (cluster_std ** 2)
        Xk = rng.multivariate_normal(means[k], cov, size=n_samples // centers)
        X.append(Xk)
        y.append(np.full(Xk.shape[0], k))
    return np.vstack(X), np.concatenate(y)


def classification_example():
    X, y = make_blobs(n_samples=900, centers=3)
    X_df = pd.DataFrame(X, columns=["x1", "x2"])
    y_sr = pd.Series(y, name="class")

    # Rede neural: 2 camadas escondidas + softmax na saída
    layers = [
        Dense(2, 32, he=True), ReLU(),
        Dense(32, 32, he=True), ReLU(),
        Dense(32, 3, he=False), Softmax()
    ]
    model = NeuralNetwork(layers, loss="cross_entropy", l2=1e-4, optimizer="adam", lr=5e-3)

    model.fit(X_df, y_sr, epochs=200, batch_size=64, val_split=0.25, patience=20, class_count=3)

    # Avaliação final
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_sr, test_size=0.25, random_state=123)
    preds = model.predict(X_test)
    acc = (preds.reshape(-1) == y_test.reshape(-1)).mean()
    print(f"Test accuracy: {acc:.3f}")


if __name__ == "__main__":
    classification_example()
