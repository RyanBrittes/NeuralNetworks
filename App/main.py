from evaluateModel import EvaluateModel

predicted_values = EvaluateModel()

pred = predicted_values.get_predict()

print(f"Accuracy Train: {pred[1]}")
