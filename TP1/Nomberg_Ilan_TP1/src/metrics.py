import numpy as np

# make a polimorphic class to implement all the metrics
class Metrics:
    def __init__(self):
        pass

    def calculate(self, y_true, y_pred):
        pass

    def __str__(self):
        pass

class MSE(Metrics):
    def __init__(self):
        super().__init__()

    def calculate(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    def __str__(self):
        return "MSE"
    
class MAE(Metrics):
    def __init__(self):
        super().__init__()

    def calculate(self, y_true, y_pred):
        return np.abs(y_true - y_pred).mean()

    def __str__(self):
        return "MAE"
    
class R2(Metrics):
    def __init__(self):
        super().__init__()

    def calculate(self, y_true, y_pred):
        return 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()

    def __str__(self):
        return "R2"