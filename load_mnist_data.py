import numpy
import idx2numpy


def load_data():
    """
    Read in the training data from the data dir.

    The data comes to us from http://yann.lecun.com/exdb/mnist/
    """
    train_images = idx2numpy.convert_from_file(
        'data/train-images-idx3-ubyte').astype('float64')

    # We know that from the IDX file documentation the max value we will be getting
    # is 255, and we wha the values of our input to be [0, 1] so
    train_images = train_images/255

    # # Lets look at one of the images
    # import matplotlib.pyplot as plt
    # plt.imsave("test", train_images[1, :, :])

    # Well that was fun

    # We also reshape the images so they match the first layer of the nn
    train_images = train_images.reshape(train_images.shape[0], 784, 1)

    # Lables
    # Along with the images we need the
    train_lables_ff = idx2numpy.convert_from_file(
        'data/train-labels-idx1-ubyte')

    # These are also in the wrong format, we need them to be activations of nodes
    train_lables = numpy.zeros([train_lables_ff.shape[0], 10, 1])

    # This cannot be the most efficient solution by it works
    for i in range(0, train_lables_ff.shape[0]):
        train_lables[i, train_lables_ff[i]] = 1

    return(train_images, train_lables)
