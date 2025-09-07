import numpy as np

class Adam():
    def get_adam(self, weights, bias, dW, dB, mW, vW, mB, vB, t, lr, beta1, beta2, delta):
        mW = beta1 * mW + (1 - beta1) * dW
        mB = beta1 * mB + (1 - beta1) * dB

        vW = beta2 * vW + (1 - beta2) * (dW**2)
        vB = beta2 * vB + (1 - beta2) * (dB**2)

        mW_adjusted = mW / (1 - beta1**t)
        mB_adjusted = mB / (1 - beta1**t)
        vW_adjusted = vW / (1 - beta2**t)
        vB_adjusted = vB / (1 - beta2**t)

        weights = weights - lr * mW_adjusted / (np.sqrt(vW_adjusted) + delta)
        bias = bias - lr * mB_adjusted / (np.sqrt(vB_adjusted) + delta)

        return [weights, bias, mW, vW, mB, vB]