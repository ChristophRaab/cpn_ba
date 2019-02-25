#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import sklearn.datasets
import nnet.nnet as nnet
from RSLVQ.rslvq import RSLVQ as nRSLVQ
from sklearn.utils.multiclass import unique_labels
from common import paths_to_tensor, load_dataset
from sklearn.utils import shuffle as unison_shuffle


class RslvqLayer(nnet.layers.Layer, nnet.layers.LossMixin):
    """
    This is an extension layer for NNET which uses RSLVQ classification. It can only occur as last layer!
    """

    def __init__(self, classAmount, stretchToOnehot=True, sigma=.2, prototypes_per_class=4, n_epochs=4):
        """
        Initializes the RslvqLayer.
        Note that one cannot set the batch size for RSLVQ since this is set to the batch size of the network.
        :param classAmount: The number of output classes.
        :param stretchToOnehot: Whether to onehot-encode the output data. Usually necessary.
        :param sigma: The sigma to use as hyperparameter.
        :param prototypes_per_class: The number. prototypes to use per input class.
        :param n_epochs: The number of epochs to learn each iteration.
        """
        self.classAmount = classAmount
        self.stretchToOnehot = stretchToOnehot
        self.init = False
        self.last_prediction = np.array([])
        self.last_input = np.array([])
        self.sigma = sigma
        self.prototypes_per_class = prototypes_per_class
        self.n_epochs = n_epochs

        #self.rslvq = nRSLVQ(sigma=.2, prototypes_per_class=4, batch_size=256, n_epochs=4)

    def _setup(self, input_shape, rng):
        """
        Build the layer. This does _not_ instantiate RSLVQ as it waits for a first batch of data in oder to call fit()
        and allow pre-initialized prototypes.
        :param input_shape: The input shape as tuple in the form (num_samples, num_inputs).
        :param rng: An instance of an random number generator.
        """
        self.n_input = input_shape[1]
        self.rng = rng

    def fprop(self, input):
        """
        Forward propagate (predict) for the given input.
        :param input: Input data as array with the shape (batch_size, num_inputs).
        :return: The prediction for this data.
        """
        self.last_input = input

        if not self.init:
            # return random data on the first run, it doesn't matter
            fitted = range(self.classAmount)
        else:
            fitted = self.rslvq.predict(input)

        self.last_prediction = fitted if not self.stretchToOnehot else self.onehot(fitted, self.classAmount)
        return self.last_prediction

    def onehot(self, fitted, nb_classes):
        """
        One-hot encodes an array.
        :param fitted: The categories to encode as number, i.e. [ 2 7 2 8 6 6 ... ].
        :param nb_classes: The maximum number of categories.
        :return: A two-dimensional array with the shape (len(fitted), nb_classes).
        """
        return np.eye(nb_classes)[fitted]

    def onehot_decode(self, data):
        """
        Reverses onehot-encoding.
        :param data: A list of onehot-encoded categories in the shape (#, num_categories).
        :return: A one-dimensional array with the fitting category for each input value.
        """
        # https://stackoverflow.com/questions/42497340/how-to-convert-one-hot-encodings-into-integers
        # edited to use 0.99999 to avoid floating point errors
        return np.array([np.where(r > 0.99999)[0][0] for r in data])

    def output_shape(self, input_shape):
        """
        Calculates the output shape of the layer given the input shape. In this layer, the input shape only determines
        the batch size; the actual output shape per sample will be determined by the number of classes and whether
        onehot-encoding is enabled.
        :param input_shape: The input shape as tuple in the form (batch_size, num_inputs).
        :return: Either (batch_size, 1) or (batch_size, num_categories).
        """
        if self.stretchToOnehot:
            return input_shape[0], self.classAmount
        return input_shape[0], 1

    def loss(self, output, output_pred):
        """ Calculate mean loss given output and predicted output. """
        # Simple loss rather than the one from the paper, Since the loss is only
        miss = np.abs(output - output_pred) / 2
        return np.mean(miss, axis=(0, 1))

    def input_grad(self, expected_output, predicted_output):
        """
        Calculate the loss, given the predicted output and the expected output. This is just a forwarding function for
        the possible ways listed below.
        :param expected_output: A onehot-encoded array in the shape (batch_size, num_categories).
        :param predicted_output: A onehot-encoded array in the shape (batch_size, num_categories).
        :return: The gradients for the next layer in the shape (batch_size, gradients_for_sample).
        """

        # our training data is the last input
        X = self.last_input

        # the output given here is the one-hot encoding of the expected one, which just happens to the correct class for
        # each data point given in last input
        y = self.onehot_decode(expected_output)

        assert(len(X.shape) == 2)
        assert(len(y.shape) == 1)

        return self.input_grad_from_learning(X, y)

    def _create_rslvq(self, batch_size):
        """
        Creates an RSLVQ instance with the parameters set in this class and a batch size.
        :return: An instance of RSLVQ.
        """
        return nRSLVQ(
            sigma=self.sigma,
            prototypes_per_class=self.prototypes_per_class,
            batch_size=batch_size,
            n_epochs=4,
            random_state=self.rng)

    def input_grad_from_closest(self, X, y):
        """
        Calculate input gradient given output and predicted output. This method works by taking the closest prototype as
        the expected value and calculating the gradient based on this. The RSLVQ implementation is also trained with the
        expected output.
        :param X: Input samples.
        :param y: The class for each input sample.
        """

        if self.init:
            rv = self.compute_rslvq_gradients_from_nearest_macthing_prototype(X, y)
            # we need to directly call _optimize since fit_batch is buggy
            self.rslvq._optimize(X, y, self.rslvq.random_state)
        else:
            xy = unique_labels(y)
            if len(xy) != self.classAmount:
                raise ValueError('The first batch did not contain a sample of every class. Unfortunately, this is '
                                 'required for initialising the RSLVQ instance. Please rerurn the program. If this '
                                 'happens frequently, try increasing the batch size.')
            self.rslvq = self._create_rslvq(y.shape[0])
            self.rslvq.fit(X, y)
            rv = self.compute_rslvq_gradients_from_nearest_macthing_prototype(X, y)
            self.init = True

        # The output value needs to be highly reduced since because the network will just feed zeros
        return rv * 0.006

    def input_grad_from_learning(self, X, y, batchwise=True):
        """
        This method calculates the output gradient from the updates to the vectors. This variant trains the network on
        either on each input sample an returns the fitting gradients or trains the network on all samples and returns
        a meaned gradient, depending on whether batchwise is set.
        :param X: Input samples.
        :param y: The class for each input sample.
        :param batchwise: Whether to do a batch fit or a fit for each sample.
        """

        if self.init:

            if batchwise:
                # save the current weights
                pre_update = self.rslvq.w_
                self.rslvq._optimize(X, y, self.rslvq.random_state)

                # calculate the diff and mean it across input dimensions
                output_grad_protos = -(self.rslvq.w_ - pre_update)
                output_grad = np.mean(output_grad_protos, axis=0)

                # repeat the gradient since we did not record individual updates
                output_grad = np.repeat(output_grad, y.shape[0], axis=0)
            else:
                # pre-allocate the gradients for each sample
                output_grad = np.zeros(X.shape)

                # fit the lvq for each sample and record the change in gradients
                for i in range(X.shape[0]):
                    _X = X[i:i+1]
                    _y = y[i:i+1]

                    # save the current weights
                    pre_update = self.rslvq.w_

                    # calculate the diff and mean it across input dimensions
                    self.rslvq._optimize(_X, _y, self.rslvq.random_state)

                    #
                    output_grad[i] = np.mean(-(self.rslvq.w_ - pre_update), axis=0)

            return output_grad * 0.006
        else:
            # there is no learning on the first pass, therefore we fall back to the simple nearest prototype learning
            xy = unique_labels(y)
            if len(xy) != self.classAmount:
                raise ValueError('The first batch did not contain a sample of every class. Unfortunately, this is '
                                 'required for initialising the RSLVQ instance. Please rerurn the program. If this '
                                 'happens frequently, try increasing the batch size.')
            self.rslvq = self._create_rslvq(y.shape[0])
            self.rslvq.fit(X, y)
            rv = self.compute_rslvq_gradients_from_nearest_macthing_prototype(X, y)
            self.init = True

            return rv * 0.006

    def compute_rslvq_gradients_from_nearest_macthing_prototype(self, X, y):
        """
        This helper computes the gradient by treating the distance to the nearest prototype with the same class as the
        expected output.
        :param X: The input in the shape (data point, input).
        :param y: The correct classes for the input.
        :return: The distance to the nearest correct prototype for each input data point
        """
        gradient = np.zeros(X.shape)
        c = 1/self.rslvq.sigma

        for k in range(X.shape[0]):
            class_id = y[k]
            point = X[k]

            prototypes_with_same_class = [self.rslvq.w_[i] for i in range(self.rslvq.w_.shape[0]) if self.rslvq.c_w_[i] == class_id]
            distance_to_same_class = [-self.rslvq._costf(point, prototypes_with_same_class[i])
                                      for i in range(len(prototypes_with_same_class))]

            min_dist = np.min(distance_to_same_class)
            min_dist_idx = np.where(distance_to_same_class == min_dist)
            assert(len(min_dist_idx) > 0)
            nearest_prototype = prototypes_with_same_class[min_dist_idx[0][0]]
            gradient[k] = (point - nearest_prototype) * c

        return gradient


class Network:
    """
    This class abstracts a nnet network to SciKit format. It also provides timings.
    """

    def __init__(self, network, learning_rate, max_iter, batch_size):
        """
        Initializes the class.
        :param network: The network to run on.
        :param learning_rate: The learning rate.
        :param max_iter: Maximum number of iterations.
        :param batch_size: The batch size.
        """
        self.network = network
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size

    def fit(self, X, y):
        """
        Trains the network with the given data.
        :param X: The input data.
        :param y: Classes for the input data
        """
        t0 = time.time()
        self.network.fit(X, y, learning_rate=self.learning_rate, max_iter=self.max_iter, batch_size=self.batch_size)
        t1 = time.time()
        print('Train duration: %.1fs' % (t1-t0))

    def predict(self, X):
        """
        Calculates the output for a given input.
        :param X: The input.
        :return: The predictions.
        """
        t0 = time.time()
        pred = self.network.predict(X)
        t1 = time.time()
        print('Predict duration: %.1fs' % (t1-t0))
        return pred

    def evaluate(self, X, y):
        """
        Evaluate the accurary of the network with the given inputs X and the expected output y.
        :param X: The input.
        :param y: The expected output.
        :return: The error rate.
        """
        t0 = time.time()
        error = self.network.error(X, y)
        t1 = time.time()
        print('Evaluat duration: %.1fs' % (t1-t0))
        print('Test error rate: %.4f' % error)
        return error


class NetworkWithRslvqLayer(Network):
    """
    This class abstracts networks with an RSLVQ layer further by correctly sorting the first input batch.
    """

    def __init__(self, network, learning_rate=0.05, max_iter=20, batch_size=256):
        """
        Initializes the class.
        :param network: The network to run on.
        :param learning_rate: The learning rate.
        param max_iter: Maximum number of iterations.
        :param batch_size: The batch size.
        """
        Network.__init__(self, network, learning_rate, max_iter, batch_size)

    def fit(self, X, y):
        """
        Trains the network with the given data.
        :param X: The input data.
        :param y: Classes for the input data
        """
        # to avoid unecessary errors we sort the data in a way that the first batch contains a sample of every class
        # this is archived by making the first [#classes] entries equal to the class at the index
        classes = unique_labels(y)

        for offset, label in enumerate(classes):
            # if the label fits already, ignore
            if y[offset] == label:
                continue

            # search the next value with the right label
            for i in range(offset + 1, len(y)):
                if y[i] == label:

                    # when found, swap positions and stop searching
                    y[i], y[offset] = y[offset], y[i]
                    X[i], X[offset] = X[offset], X[i]
                    break

        Network.fit(self, X, y)


class ConvRslvqNetwork(NetworkWithRslvqLayer):
    """
    Big convoluted network for classifying dog breed data.
    """

    def __init__(self, n_classes, learning_rate=0.05, max_iter=20, batch_size=256):
        """
        Initializes the class.
        :param n_classes: The number of output classes. Must be smaller or equal to the batch size.
        :param learning_rate: The learning rate.
        :param max_iter: Maximum number of iterations.
        :param batch_size: The batch size.
        """
        if batch_size < n_classes:
            raise ValueError('The batch size must be at least as big as the number of classes since the first training '
                             'batch must contain a sample of every class!')

        # Setup convolutional neural network
        nn = nnet.NeuralNetwork(
            layers=[
                nnet.Conv(
                    n_feats=16,
                    filter_shape=(3, 3),
                    strides=(1, 1),
                    weight_scale=1,
                    weight_decay=0.001,
                ),
                nnet.Activation('relu'),
                nnet.Pool(
                    pool_shape=(2, 2),
                    strides=(2, 2),
                    mode='max',
                ),

                nnet.Conv(
                    n_feats=32,
                    filter_shape=(3, 3),
                    strides=(1, 1),
                    weight_scale=1,
                    weight_decay=0.001,
                ),
                nnet.Activation('relu'),
                nnet.Pool(
                    pool_shape=(2, 2),
                    strides=(2, 2),
                    mode='max',
                ),

                nnet.Conv(
                    n_feats=64,
                    filter_shape=(3, 3),
                    strides=(1, 1),
                    weight_scale=1,
                    weight_decay=0.001,
                ),
                nnet.Activation('relu'),
                nnet.Pool(
                    pool_shape=(2, 2),
                    strides=(2, 2),
                    mode='max',
                ),

                nnet.Conv(
                    n_feats=128,
                    filter_shape=(3, 3),
                    strides=(1, 1),
                    weight_scale=1,
                    weight_decay=0.001,
                ),
                nnet.Activation('relu'),
                nnet.Pool(
                    pool_shape=(2, 2),
                    strides=(2, 2),
                    mode='max',
                ),

                nnet.Flatten(),
                RslvqLayer(n_classes, stretchToOnehot=True),
            ],
        )

        NetworkWithRslvqLayer.__init__(self, nn, learning_rate, max_iter, batch_size)


class DogImageNetworkComparison(Network):
    """
    Big convoluted network for classifying dog breed data.
    """

    def __init__(self, n_classes, learning_rate=0.05, max_iter=20, batch_size=256):
        """
        Initializes the class.
        :param n_classes: The number of output classes. Must be smaller or equal to the batch size.
        :param learning_rate: The learning rate.
        :param max_iter: Maximum number of iterations.
        :param batch_size: The batch size.
        """
        if batch_size < n_classes:
            raise ValueError('The batch size must be at least as big as the number of classes since the first training '
                             'batch must contain a sample of every class!')

        # Setup convolutional neural network
        nn = nnet.NeuralNetwork(
            layers=[
                nnet.Conv(
                    n_feats=16,
                    filter_shape=(3, 3),
                    strides=(1, 1),
                    weight_scale=1,
                    weight_decay=0.001,
                ),
                nnet.Activation('relu'),
                nnet.Pool(
                    pool_shape=(2, 2),
                    strides=(2, 2),
                    mode='max',
                ),

                nnet.Conv(
                    n_feats=32,
                    filter_shape=(3, 3),
                    strides=(1, 1),
                    weight_scale=1,
                    weight_decay=0.001,
                ),
                nnet.Activation('relu'),
                nnet.Pool(
                    pool_shape=(2, 2),
                    strides=(2, 2),
                    mode='max',
                ),

                nnet.Conv(
                    n_feats=64,
                    filter_shape=(3, 3),
                    strides=(1, 1),
                    weight_scale=1,
                    weight_decay=0.001,
                ),
                nnet.Activation('relu'),
                nnet.Pool(
                    pool_shape=(2, 2),
                    strides=(2, 2),
                    mode='max',
                ),

                nnet.Conv(
                    n_feats=128,
                    filter_shape=(3, 3),
                    strides=(1, 1),
                    weight_scale=1,
                    weight_decay=0.001,
                ),
                nnet.Activation('relu'),
                nnet.Pool(
                    pool_shape=(2, 2),
                    strides=(2, 2),
                    mode='max',
                ),

                nnet.Flatten(),

            ],
        )

        Network.__init__(self, nn, learning_rate, max_iter, batch_size)


def run_dogbreed():
    """
    Runs the network on the dogbreed data set.
    """
    # load train, test, and validation datasets and merge train+valid since this nn does not use validation
    train_files, train_targets = load_dataset('CodeData/dogImages/train', onehot=False)
    valid_files, valid_targets = load_dataset('CodeData/dogImages/valid', onehot=False)
    test_files, y_test = load_dataset('CodeData/dogImages/test', onehot=False)

    train_files = np.concatenate((train_files, valid_files), axis=0)
    y_train = np.concatenate((train_targets, valid_targets), axis=0)

    # shuffle the training data
    train_files, y_train = unison_shuffle(train_files, y_train)

    # load the images and pre-process the data for the network by normalizing it
    # zscore
    X_train = paths_to_tensor(train_files, data_format='channels_first').astype('float64') / 255
    X_test = paths_to_tensor(test_files, data_format='channels_first').astype('float64') / 255
    n_classes = np.unique(y_train).size

    # create the network
    nn = ConvRslvqNetwork(n_classes)

    # Train neural network
    nn.fit(X_train, y_train)

    # Evaluate on test data
    error = nn.evaluate(X_test, y_test)
    print('Test error rate: %.4f' % error)

def run_mnist():
    """
    Runs the network on the MNIST data set.
    """
    # Fetch data
    mnist = sklearn.datasets.fetch_mldata('MNIST original')
    split = 60000
    X_train = np.reshape(mnist.data[:split], (-1, 1, 28, 28))/255.0
    y_train = mnist.target[:split].astype("int")
    X_test = np.reshape(mnist.data[split:], (-1, 1, 28, 28))/255.0
    y_test = mnist.target[split:].astype("int")
    n_classes = np.unique(y_train).size

    # Downsample training data
    n_train_samples = 3000
    train_idxs = np.random.random_integers(0, split-1, n_train_samples)
    X_train = X_train[train_idxs, ...]
    y_train = y_train[train_idxs, ...]

    # Setup convolutional neural network
    nn = nnet.NeuralNetwork(
        layers=[
            nnet.Conv(
                n_feats=12,
                filter_shape=(5, 5),
                strides=(1, 1),
                weight_scale=0.1,
                weight_decay=0.001,
            ),
            nnet.Activation('relu'),
            nnet.Pool(
                pool_shape=(2, 2),
                strides=(2, 2),
                mode='max',
            ),
            nnet.Conv(
                n_feats=16,
                filter_shape=(5, 5),
                strides=(1, 1),
                weight_scale=0.1,
                weight_decay=0.004,
            ),
            nnet.Activation('relu'), # softmax?
            nnet.Flatten(),
            RslvqLayer(n_classes, stretchToOnehot=True), # zscore
        ],
    )

    # Train neural network
    t0 = time.time()
    nn.fit(X_train, y_train, learning_rate=0.05, max_iter=20, batch_size=64)
    t1 = time.time()
    print('Duration: %.1fs' % (t1-t0))

    # Evaluate on test data
    error = nn.error(X_test, y_test)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run_dogbreed()
