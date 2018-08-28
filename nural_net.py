import math
import numpy

import load_mnist_data

train_images, train_lables = load_mnist_data.load_data()


def sigmoid(z):
    """
    Apply the sigmoid function to a given vector z

    Keyword arguments:
    z -- numpy vector of values to apply function to
    """
    return 1/(1 + numpy.exp(z))


def sigmoid_prime(z):
    """
    Apply the derivative of the sigmoid function to a given vector z

    Keyword arguments:
    z -- numpy vector of values to apply function to
    """
    sig_z = sigmoid(z)
    return sig_z * (1 - sig_z)

# Define a network
# We take the __init__ of a network from:
# http://neuralnetworksanddeeplearning.com/chap1.html
# We will try and work out the rest ourselves


class Network(object):

    def __init__(self, sizes, learning_rate):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [numpy.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def run_network(self, x):
        """
        Run the network with the first layer of nodes in state x

        Keyword arguments:
        x -- numpy vector of start conditions of the initial node
        """
        # We have the input vector x, this is the first layer in our nn
        a = x

        # So we need to run the neural network now.  We are going to make
        # things extra explicit by having the z term.  Clearly super
        # inefficient
        for w, b in zip(self.weights, self.biases):
            # Find the values in the next layer of the network
            z = numpy.add(numpy.dot(w, a), b)
            a = sigmoid(z)

        return a

    def backpropagation(self, x, y):
        """
        Apply the backpropagation algorithm to the network.  This will be
        conducted using just one observation.

        Keyword arguments:
        x -- numpy vector of start conditions of the initial node
        y -- the correct values of the end nodes

        We will:
        1) Set the first layer a_1 tp x_1
        2) Feedforward
        3) Calculate the output error
        4) Backpropagate the error
        5) Calculate the gradient of the cost function
        """
        # 1) Set the initial (probably not needed will leave in for now for
        # notational simplicity)
        a = [x]
        z = []
        output_error = [numpy.zeros(bias.shape) for bias in self.biases]
        par_w_by_c = [numpy.zeros(weights.shape) for weights in self.weights]

        # 2) Feedforward
        for w, b in zip(self.weights, self.biases):
            # Find the values in the next layer of the network
            z.append(numpy.add(numpy.dot(w, a), b))
            a.append(sigmoid(z[-1]))

        # 3) Output error
        output_error[-1] = (a[-1] - y) * sigmoid_prime(z[-1])

        # 4) Backpropagate the error
        for i in range(self.num_layers - 2, 0, -1):

            # Calculate the output error for the layer
            output_error[i] = numpy.dot(
                self.weights[i].transpose(),
                output_error[i + 1]) * sigmoid_prime(z[i])

            par_w_by_c[i] = numpy.dot([output_error[i]],
                                      a[[i-1]].transpose())

        return output_error, par_w_by_c

    def single_gradient_descent(self, x, y):
        # Run a back propagation
        output_error, par_w_by_c = self.backpropagation(x, y)

        # And the gradient decent
        return 13


# Let us get one of those networks
net = Network([784, 16, 20, 10], 1)

net.run_network(train_images[1123])

net.backpropagation(train_images[1], train_lables[1])

# for i, (image, label) in enumerate(zip(train_images, train_lables)):
#    net.single_backpropagation(image, label)
#    if i % 1000 == 0:
#        print(i)


# Plot the image
import matplotlib.pyplot as plt
id = 103

test_image = train_images[id].reshape(1, 28, 28)
plt.imsave("test", test_image[0])
