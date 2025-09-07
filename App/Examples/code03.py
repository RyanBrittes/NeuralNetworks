import numpy as np
import pandas as pd

# -----------------------------
# 1. Carregar dados
# -----------------------------
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1).values
y = data["Outcome"].values.reshape(1, -1)

# Normalizar (média 0, desvio 1)
X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

# Transpor X para (n_features, m_examples)
X = X.T

# Dividir treino e teste
m = X.shape[1]
m_train = int(m * 0.8)
indices = np.random.permutation(m)

X_train, X_test = X[:, indices[:m_train]], X[:, indices[m_train:]]
y_train, y_test = y[:, indices[:m_train]], y[:, indices[m_train:]]

# -----------------------------
# 2. Funções auxiliares
# -----------------------------
def sigmoid(Z): return 1 / (1 + np.exp(-Z))
def relu(Z): return np.maximum(0, Z)
def relu_backward(dA, Z): return dA * (Z > 0)

def compute_loss(y, y_hat):
    m = y.shape[1]
    return -(1/m) * np.sum(y*np.log(y_hat+1e-8) + (1-y)*np.log(1-y_hat+1e-8))

# -----------------------------
# 3. Inicializar parâmetros
# -----------------------------
def initialize_params(n_x, n_h, n_y):
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * np.sqrt(2/n_x)  # He init
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * np.sqrt(2/n_h)
    b2 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# -----------------------------
# 4. Forward
# -----------------------------
def forward(X, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# -----------------------------
# 5. Backward
# -----------------------------
def backward(X, y, params, cache):
    m = X.shape[1]
    W2 = params["W2"]
    A1, A2 = cache["A1"], cache["A2"]
    Z1 = cache["Z1"]

    dZ2 = A2 - y
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T @ dZ2
    dZ1 = relu_backward(dA1, Z1)
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

# -----------------------------
# 6. Update
# -----------------------------
def update(params, grads, lr=0.01):
    params["W1"] -= lr * grads["dW1"]
    params["b1"] -= lr * grads["db1"]
    params["W2"] -= lr * grads["dW2"]
    params["b2"] -= lr * grads["db2"]
    return params

# -----------------------------
# 7. Train
# -----------------------------
def train(X, y, n_h=16, epochs=1000, lr=0.01):
    n_x, n_y = X.shape[0], y.shape[0]
    params = initialize_params(n_x, n_h, n_y)
    for i in range(epochs):
        A2, cache = forward(X, params)
        loss = compute_loss(y, A2)
        grads = backward(X, y, params, cache)
        params = update(params, grads, lr)
        if i % 100 == 0:
            print(f"Epoch {i} - Loss: {loss:.4f}")
    return params

# -----------------------------
# 8. Predição e Acurácia
# -----------------------------
def predict(X, y, params):
    A2, _ = forward(X, params)
    preds = (A2 > 0.5).astype(int)
    acc = np.mean(preds == y)
    return preds, acc

# -----------------------------
# 9. Executar
# -----------------------------
params = train(X_train, y_train, n_h=16, epochs=1000, lr=0.01)

_, acc_train = predict(X_train, y_train, params)
_, acc_test = predict(X_test, y_test, params)

print("Acurácia Treino:", acc_train)
print("Acurácia Teste :", acc_test)
