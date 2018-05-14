import numpy as np


class InputLayer:
    def __init__(self, neuroNum):
        self.neuroNum = neuroNum
        self.data = np.zeros((1, self.neuroNum))


class NeuroLayer:
    def __init__(self, neuroNum, preLayer, bias):
        self.neuroNum = neuroNum
        self.preLayer = preLayer
        self.data = np.zeros((1, self.neuroNum))
        r = np.sqrt(6 / (self.neuroNum + self.preLayer.neuroNum))
        self.weight = np.random.uniform(-r, r, (self.neuroNum, self.preLayer.neuroNum))
        self.bias = np.zeros((1, self.neuroNum))
        self.bias.fill(bias)
        self.nextLayer = None
        self.preLayer.nextLayer = self
        self.diff = np.zeros((1, self.preLayer.neuroNum))
        self.diffWeight = np.zeros((self.neuroNum, self.preLayer.neuroNum))
        self.diffBias = np.zeros((1, self.neuroNum))

    def forward(self):
        temp = np.dot(self.preLayer.data, self.weight.T)
        self.data = temp + self.bias

    def backward(self):
        self.diffWeight += np.dot(self.nextLayer.diff.T, self.preLayer.data)
        self.diffBias += self.nextLayer.diff * 1
        self.diff = np.dot(self.nextLayer.diff, self.weight)

    def update(self, lr):
        self.bias -= self.diffBias * lr
        self.weight -= self.diffWeight * lr
        self.diffBias = np.zeros((1, self.neuroNum))
        self.diffWeight = np.zeros((self.neuroNum, self.preLayer.neuroNum))


class Sigmoid:
    def __init__(self, preLayer):
        self.preLayer = preLayer
        self.neuroNum = self.preLayer.neuroNum
        self.data = np.zeros((1, self.preLayer.neuroNum))
        self.nextLayer = None
        self.preLayer.nextLayer = self
        self.diff = np.zeros((1, self.preLayer.neuroNum))

    def activate(self, x):
        return np.ones(self.neuroNum) / (np.ones(self.neuroNum) + np.exp(-x))

    def derivation(self, y):
        return y * (np.ones(self.neuroNum) - y)

    def forward(self):
        self.data = self.activate(self.preLayer.data)

    def backward(self):
        self.diff = self.nextLayer.diff * self.derivation(self.data)

    def update(self, lr):
        pass


class PRelu:
    def __init__(self, preLayer):
        self.preLayer = preLayer
        self.neuroNum = self.preLayer.neuroNum
        self.data = np.zeros((1, self.preLayer.neuroNum))
        self.nextLayer = None
        self.preLayer.nextLayer = self
        self.diff = np.zeros((1, self.preLayer.neuroNum))

    def activate(self, x):
        return np.where(x < 0, x * 0.25, x)

    def derivation(self, y):
        return np.where(y < 0, 0.25, 1)

    def forward(self):
        self.data = self.activate(self.preLayer.data)

    def backward(self):
        self.diff = self.nextLayer.diff * self.derivation(self.preLayer.data)

    def update(self, lr):
        pass


class ErrorLayer:
    def __init__(self, preLayer):
        self.preLayer = preLayer
        self.neuroNum = self.preLayer.neuroNum
        self.data = 0.0
        self.target = np.zeros((1, self.neuroNum))
        self.diff = np.zeros((1, self.preLayer.neuroNum))
        self.preLayer.nextLayer = self

    def forward(self):
        self.data += np.power(self.preLayer.data - self.target, 2) * 0.5

    def backward(self):
        self.diff = self.preLayer.data - self.target

    def update(self, lr):
        self.data = 0.0
