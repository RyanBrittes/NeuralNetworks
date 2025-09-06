import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    Implementação de uma Rede Neural Multi-Layer Perceptron usando apenas NumPy
    """
    
    def __init__(self, layers: List[int], learning_rate: float = 0.01, random_seed: int = 42):
        """
        Inicializa a rede neural
        
        Args:
            layers: Lista com o número de neurônios em cada camada [input, hidden1, hidden2, ..., output]
            learning_rate: Taxa de aprendizado
            random_seed: Seed para reprodutibilidade
        """
        np.random.seed(random_seed)
        self.layers = layers
        self.learning_rate = learning_rate
        self.num_layers = len(layers)
        
        # Inicialização dos pesos usando Xavier/Glorot
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Xavier initialization
            weight = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            bias = np.zeros((1, layers[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
        
        # Histórico de treinamento
        self.loss_history = []
        self.accuracy_history = []
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Função de ativação Sigmoid"""
        # Evita overflow clipping os valores
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivada da função Sigmoid"""
        return z * (1 - z)
    
    def relu(self, z: np.ndarray) -> np.ndarray:
        """Função de ativação ReLU"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivada da função ReLU"""
        return (z > 0).astype(float)
    
    def softmax(self, z: np.ndarray) -> np.ndarray:
        """Função Softmax para classificação multi-classe"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X: np.ndarray, activation_function: str = 'sigmoid') -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation através da rede
        
        Args:
            X: Dados de entrada (m, n_features)
            activation_function: 'sigmoid' ou 'relu'
        
        Returns:
            activations: Lista com as ativações de cada camada
            z_values: Lista com os valores z (antes da ativação) de cada camada
        """
        activations = [X]  # A primeira ativação é a entrada
        z_values = []
        
        for i in range(self.num_layers - 1):
            # Calcula z = W*a + b
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Aplica função de ativação
            if i == self.num_layers - 2:  # Última camada
                if self.layers[-1] == 1:  # Classificação binária
                    a = self.sigmoid(z)
                else:  # Classificação multi-classe
                    a = self.softmax(z)
            else:  # Camadas ocultas
                if activation_function == 'sigmoid':
                    a = self.sigmoid(z)
                else:  # relu
                    a = self.relu(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula a função de perda (cross-entropy)
        """
        m = y_true.shape[0]
        
        if self.layers[-1] == 1:  # Classificação binária
            loss = -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        else:  # Classificação multi-classe
            loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
        
        return loss
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                           z_values: List[np.ndarray], activation_function: str = 'sigmoid') -> None:
        """
        Backward propagation e atualização dos pesos
        """
        m = X.shape[0]
        
        # Calcula o erro da última camada
        if self.layers[-1] == 1:  # Classificação binária
            delta = activations[-1] - y
        else:  # Classificação multi-classe
            delta = activations[-1] - y
        
        # Backpropagation através das camadas
        for i in range(self.num_layers - 2, -1, -1):
            # Calcula gradientes
            dW = np.dot(activations[i].T, delta) / m
            db = np.mean(delta, axis=0, keepdims=True)
            
            # Atualiza pesos e biases
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            
            # Calcula delta para a camada anterior (se não for a primeira)
            if i > 0:
                if activation_function == 'sigmoid':
                    delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])
                else:  # relu
                    delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(z_values[i-1])
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              activation_function: str = 'sigmoid', verbose: bool = True) -> None:
        """
        Treina a rede neural
        
        Args:
            X: Dados de treinamento (m, n_features)
            y: Labels (m, n_classes) para multi-classe ou (m, 1) para binária
            epochs: Número de épocas
            activation_function: 'sigmoid' ou 'relu'
            verbose: Se deve imprimir o progresso
        """
        for epoch in range(epochs):
            # Forward propagation
            activations, z_values = self.forward_propagation(X, activation_function)
            
            # Calcula a perda
            loss = self.compute_loss(y, activations[-1])
            self.loss_history.append(loss)
            
            # Calcula acurácia
            accuracy = self.calculate_accuracy(y, activations[-1])
            self.accuracy_history.append(accuracy)
            
            # Backward propagation
            self.backward_propagation(X, y, activations, z_values, activation_function)
            
            # Imprime progresso
            if verbose and epoch % 100 == 0:
                print(f'Época {epoch:4d} | Perda: {loss:.6f} | Acurácia: {accuracy:.4f}')
    
    def predict(self, X: np.ndarray, activation_function: str = 'sigmoid') -> np.ndarray:
        """
        Faz predições
        
        Returns:
            Predições (probabilidades para multi-classe, 0/1 para binária)
        """
        activations, _ = self.forward_propagation(X, activation_function)
        predictions = activations[-1]
        
        if self.layers[-1] == 1:  # Classificação binária
            return (predictions > 0.5).astype(int)
        else:  # Classificação multi-classe
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray, activation_function: str = 'sigmoid') -> np.ndarray:
        """
        Retorna as probabilidades das predições
        """
        activations, _ = self.forward_propagation(X, activation_function)
        return activations[-1]
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula a acurácia
        """
        if self.layers[-1] == 1:  # Classificação binária
            predictions = (y_pred > 0.5).astype(int)
            return np.mean(predictions == y_true)
        else:  # Classificação multi-classe
            predictions = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(y_true, axis=1)
            return np.mean(predictions == y_true_labels)
    
    def plot_training_history(self) -> None:
        """
        Plota o histórico de treinamento
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.loss_history)
        ax1.set_title('Perda durante o Treinamento')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Perda')
        ax1.grid(True)
        
        ax2.plot(self.accuracy_history)
        ax2.set_title('Acurácia durante o Treinamento')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Acurácia')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


# Exemplo de uso da rede neural

def create_sample_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria dados de exemplo para classificação binária (XOR problem)
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    # XOR: y = 1 se x1*x2 > 0, senão y = 0
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int).reshape(-1, 1)
    return X, y

def create_multiclass_data(n_samples: int = 1000, n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria dados de exemplo para classificação multi-classe
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    
    # Cria classes baseadas em regiões circulares
    distances = np.linalg.norm(X, axis=1)
    y_labels = np.digitize(distances, bins=np.linspace(0, 3, n_classes)) - 1
    y_labels = np.clip(y_labels, 0, n_classes - 1)
    
    # One-hot encoding
    y = np.eye(n_classes)[y_labels]
    
    return X, y

# Exemplo de uso
if __name__ == "__main__":
    print("=== Exemplo 1: Classificação Binária (Problema XOR) ===")
    
    # Cria dados
    X_bin, y_bin = create_sample_data(1000)
    
    # Divide em treino e teste
    train_size = int(0.8 * len(X_bin))
    X_train, X_test = X_bin[:train_size], X_bin[train_size:]
    y_train, y_test = y_bin[:train_size], y_bin[train_size:]
    
    # Cria e treina a rede neural para classificação binária
    # Arquitetura: 2 neurônios de entrada, 4 na camada oculta, 1 de saída
    nn_binary = NeuralNetwork(layers=[2, 4, 1], learning_rate=0.1)
    nn_binary.train(X_train, y_train, epochs=1000, activation_function='sigmoid')
    
    # Avalia o modelo
    train_accuracy = nn_binary.calculate_accuracy(y_train, nn_binary.predict_proba(X_train))
    test_accuracy = nn_binary.calculate_accuracy(y_test, nn_binary.predict_proba(X_test))
    
    print(f"Acurácia no treino: {train_accuracy:.4f}")
    print(f"Acurácia no teste: {test_accuracy:.4f}")
    
    print("\n=== Exemplo 2: Classificação Multi-classe ===")
    
    # Cria dados multi-classe
    X_multi, y_multi = create_multiclass_data(1000, n_classes=3)
    
    # Divide em treino e teste
    X_train_multi, X_test_multi = X_multi[:train_size], X_multi[train_size:]
    y_train_multi, y_test_multi = y_multi[:train_size], y_multi[train_size:]
    
    # Cria e treina a rede neural para classificação multi-classe
    # Arquitetura: 2 neurônios de entrada, 5 na camada oculta, 3 de saída
    nn_multi = NeuralNetwork(layers=[2, 5, 3], learning_rate=0.1)
    nn_multi.train(X_train_multi, y_train_multi, epochs=1000, activation_function='relu')
    
    # Avalia o modelo
    train_accuracy_multi = nn_multi.calculate_accuracy(y_train_multi, nn_multi.predict_proba(X_train_multi))
    test_accuracy_multi = nn_multi.calculate_accuracy(y_test_multi, nn_multi.predict_proba(X_test_multi))
    
    print(f"Acurácia no treino: {train_accuracy_multi:.4f}")
    print(f"Acurácia no teste: {test_accuracy_multi:.4f}")
    
    # Exemplo com dados do pandas
    print("\n=== Exemplo 3: Usando Pandas DataFrame ===")
    
    # Converte para DataFrame do pandas
    columns = [f'feature_{i}' for i in range(X_bin.shape[1])]
    df = pd.DataFrame(X_bin, columns=columns)
    df['target'] = y_bin.ravel()
    
    print("Primeiras linhas do dataset:")
    print(df.head())
    
    # Treina usando dados do pandas
    X_pandas = df[columns].values
    y_pandas = df['target'].values.reshape(-1, 1)
    
    nn_pandas = NeuralNetwork(layers=[2, 6, 1], learning_rate=0.05)
    nn_pandas.train(X_pandas, y_pandas, epochs=500, verbose=False)
    
    accuracy_pandas = nn_pandas.calculate_accuracy(y_pandas, nn_pandas.predict_proba(X_pandas))
    print(f"Acurácia com dados do Pandas: {accuracy_pandas:.4f}")