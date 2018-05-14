from net import InputLayer, NeuroLayer, PRelu, ErrorLayer, Sigmoid
import time
import numpy as np
from visdom import Visdom


def main():
    start_time = time.clock()
    lr = 0.0000006  # 学习率
    iteration = 10000  # 迭代次数
    bias = 0.5  # 初始化bias
    testNum = 30  # 测试集数量

    viz = Visdom(env='x1+x2')

    trainingData = np.random.uniform(-100, 100, (30, 1, 2))
    testData = np.random.uniform(-500, 500, (testNum, 1, 2))
    rightResult = testData[:, :, 0] + testData[:, :, 1]
    rightDot = np.append(testData[:, 0, :], rightResult, 1)
    neuralNetwork = []
    inputLayer = InputLayer(len(trainingData[0, 0]))
    h1 = NeuroLayer(5, inputLayer, bias)
    neuralNetwork.append(h1)
    a1 = PRelu(h1)
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
            inputLayer.data = d
            errorLayer.target = [d[0, 0] + d[0, 1]]
            for layer in neuralNetwork:
                layer.forward()
            for layer in reversed(neuralNetwork):
                layer.backward()
            last_error += errorLayer.data
            for layer in neuralNetwork:
                layer.update(lr)

        if(100 > itr > 19 or itr % 20 == 0):
            if(itr == 20):
                win = viz.line(
                    X=np.array([itr]),
                    Y=np.array(last_error[0] / len(trainingData)),
                    name="x1+x2",
                    win='loss'
                )
                win2 = viz.scatter(
                    X=np.random.rand(1, 2),
                    name="x1+x2 dot",
                    win='fitting',
                )
            viz.updateTrace(
                X=np.array([itr]),
                Y=np.array(last_error[0] / len(trainingData)),
                win=win,
            )
            testResult = np.empty([testNum, 1])
            for i in range(len(testData)):
                inputLayer.data = testData[i]
                for layer in neuralNetwork[:-1]:
                    layer.forward()
                testResult[i][0] = outputActionLayer.data[0][0]
            testDot = np.append(testData[:, 0, :], testResult, 1)
            dot = np.append(rightDot, testDot, 0)
            viz.scatter(
                X=dot,
                name="x1+x2 dot",
                win=win2,
                Y=[1] * testNum + [2] * testNum,
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
        inputLayer.data = d
        errorLayer.target = [d[0, 0] + d[0, 1]]
        for layer in neuralNetwork:
            layer.forward()
        last_error += errorLayer.data
        print('_______________')
        print(d, [d[0, 0] + d[0, 1]])
        print("output", outputActionLayer.data)
        print("error", errorLayer.data)
        print('_______________')
    print('last_error', last_error / len(testData))


if __name__ == '__main__':
    main()
