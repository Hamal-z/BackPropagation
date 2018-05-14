from net import InputLayer, NeuroLayer, PRelu, ErrorLayer, Sigmoid
import time
import numpy as np
from visdom import Visdom


def main():
    start_time = time.clock()
    lr = 0.0008
    iteration = 10000
    bias = 0.5

    viz = Visdom(env='x**2')
    trainingData = np.empty([0, 1, 1])
    for i in range(40):
        a = i * 25 / 40
        trainingData = np.append(trainingData, [[[a]]], 0)
    testData = np.random.uniform(0, 25, (30, 1, 1))
    rightResult = np.square(testData)
    rightDot = np.append(testData[:, :, 0], rightResult[:, :, 0], 1)

    neuralNetwork = []
    inputLayer = InputLayer(len(trainingData[0, 0]))
    h1 = NeuroLayer(5, inputLayer, bias)
    neuralNetwork.append(h1)
    a1 = Sigmoid(h1)
    neuralNetwork.append(a1)
    outputLayer = NeuroLayer(1, a1, bias)
    outputActionLayer = PRelu(outputLayer)
    errorLayer = ErrorLayer(outputActionLayer)
    neuralNetwork.append(outputLayer)
    neuralNetwork.append(outputActionLayer)
    neuralNetwork.append(errorLayer)

    init_weight = h1.weight.copy()
    for itr in range(1, iteration):
        np.random.shuffle(trainingData)
        last_error = 0
        for d in trainingData:
            inputLayer.data = [d[0] / 10]
            errorLayer.target = [d[0, 0] * d[0, 0] / 100]
            for layer in neuralNetwork:
                layer.forward()
            for layer in reversed(neuralNetwork):
                layer.backward()
            last_error += errorLayer.data * 10000
            for layer in neuralNetwork:
                layer.update(lr)

        if(100 > itr > 19 or itr % 20 == 0):
            if(itr == 20):
                win = viz.line(
                    X=np.array([itr]),
                    Y=np.array(last_error[0] / len(trainingData)),
                    name="x**2",
                    win='loss'
                )
                win2 = viz.scatter(
                    X=np.random.rand(1, 2),
                    name="x2 dot",
                    win='fitting',
                )
            viz.updateTrace(
                X=np.array([itr]),
                Y=np.array(last_error[0] / len(trainingData)),
                win=win,
            )

            testResult = np.empty([30, 1])
            for i in range(len(testData)):
                inputLayer.data = testData[i] / 10
                for layer in neuralNetwork[:-1]:
                    layer.forward()
                testResult[i][0] = outputActionLayer.data[0][0] * 100
            testDot = np.append(testData[:, :, 0], testResult, 1)
            dot = np.append(rightDot, testDot, 0)
            viz.scatter(
                X=dot,
                name="x2 dot",
                win=win2,
                Y=[1] * 30 + [2] * 30,
                opts=dict(
                    legend=['right', 'test'],
                    markersize=5,
                )
            )

    print('=================================')
    print('last_error', last_error)
    elapsed_time = time.clock() - start_time
    print("all time", elapsed_time)
    print('=================================')
    print('init weight', init_weight)
    print('last weight', h1.weight)
    print('=================================')

    last_error = 0
    for d in testData:
        inputLayer.data = [d[0] / 10]
        errorLayer.target = [d[0, 0] * d[0, 0] / 100]
        for layer in neuralNetwork:
            layer.forward()
        last_error += errorLayer.data * 10000
        print('_______________')
        print(d, [d[0, 0] * d[0, 0]])
        print("output", outputActionLayer.data * 100)
        print("error", errorLayer.data * 10000)
        print('_______________')
        errorLayer.update(0)
    print('last_error', last_error / len(testData))


if __name__ == '__main__':
    main()
