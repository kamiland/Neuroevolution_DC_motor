import numpy as np


def sigmoid(x, return_derivative=''):
    if 'derivative' == return_derivative:
        return x * (1.0 - x)
    else:
        return 1.0 / (1 + np.exp(-x))


class NeuralNet:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backpropagation(self):
        new_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid(self.output, 'derivative')))
        new_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid(self.output, 'derivative'),
                                                    self.weights2.T) * sigmoid(self.layer1, 'derivative')))

        self.weights1 += new_weights1
        self.weights2 += new_weights2


if __name__ == '__main__':
    print('Hello, this is simple neural network test.')

    X = np.array([[0.0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    y = np.array([[0], [1], [1], [0.3]])

    print('Input:')
    print(X)

    print('Desired output:')
    print(y.T)

    nn = NeuralNet(X, y)

    for i in range(1500):
        nn.feedforward()
        nn.backpropagation()

    output = nn.output
    print('Trained neural network output:')
    print(output.T)

    # print('Weights Vector 1:\n', nn.weights1)
    # print('Weights Vector 2:\n', nn.weights1)

    # Test on another input values
    while 1:
        different_X = np.copy(X)
        deviation_range = float(input("Give deviation value (i.e. 0.1): "))

        for i in range(len(different_X)):
            for j in range(len(different_X[i])):
                # get a random value from < -deviation_range/2 ; deviation_range/2 )
                different_X[i][j] += (deviation_range * (np.random.rand() - 0.5))

        print('Different input, (each element of initial input + value from <-%.2f,%.2f>):'
              % (deviation_range / 2, deviation_range / 2))
        print(different_X)

        nn.input = different_X
        nn.feedforward()
        new_output = nn.output
        print('Output of the same Neural Net with different input:')
        print(new_output.T)

        print('Relative error [%]:')
        print((np.divide(np.subtract(output, new_output), output) * 100).T)