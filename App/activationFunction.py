from reLU import ReLU
from sigmoid import Sigmoid
from tanh import Tanh

class ActivationFunction():
    def __init__(self):
        self.reLU = ReLU()
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def get_function(self, func: str):
        chosen_function = ""
        if func == "reLU":
            chosen_function = self.reLU
        if func == "sigmoid":
            chosen_function = self.sigmoid
        if func == "tanh":
            chosen_function = self.tanh
        else:
            print("Função de ativação não reconhecida")
            chosen_function = False
        return chosen_function