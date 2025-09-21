
class GradientDescent():

    def calc_gradient(self, parameters, grads, lr):
        parameters["weights_01"] -= lr * grads["d_weights_01"]
        parameters["bias_01"] -= lr * grads["d_bias_01"]
        parameters["weights_02"] -= lr * grads["d_weights_02"]
        parameters["bias_02"] -= lr * grads["d_bias_02"]

        return parameters
