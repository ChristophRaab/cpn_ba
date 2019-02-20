#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import sklearn.datasets
import nnet.nnet as nnet
#from modified_rslvq import RSLVQ as nRSLVQ
from RSLVQ.rslvq import RSLVQ as nRSLVQ
from sklearn.utils.multiclass import unique_labels
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class RslvqLayer(nnet.layers.Layer, nnet.layers.LossMixin):
    def __init__(self, classAmount, stretchToOnehot, rslvq=None):
        self.rslvq = rslvq
        self.classAmount = classAmount
        self.stretchToOnehot = stretchToOnehot
        self.init = False
        self.rslvq = nRSLVQ(sigma=.2, prototypes_per_class=4, batch_size=256, n_epochs=4)
        self.last_prediction = np.array([])
        self.last_input = np.array([])

    def _setup(self, input_shape, rng):
        self.n_input = input_shape[1]
        self.rng = rng

    def fprop(self, input):
        self.last_input = input

        if not self.init:
            # return random data on the first run, it doesn't matter
            fitted = range(self.classAmount)
        else:
            fitted = self.rslvq.predict(input)

        self.last_prediction = fitted if not self.stretchToOnehot else self.onehot(fitted, self.classAmount)
        return self.last_prediction

    def onehot(self, fitted, nb_classes):
        return np.eye(nb_classes)[fitted]

    def onehot_decode(self, data):
        # https://stackoverflow.com/questions/42497340/how-to-convert-one-hot-encodings-into-integers
        # edited to use 0.99999 to avoid floating point errors
        return np.array([np.where(r > 0.99999)[0][0] for r in data])

    def output_shape(self, input_shape):
        if self.stretchToOnehot:
            return input_shape[0], self.classAmount
        return input_shape[0], 1

    def loss(self, output, output_pred):
        """ Calculate mean loss given output and predicted output. """
        # rather simple loss rather than the one from the paper
        miss = np.abs(output - output_pred) / 2
        return np.mean(miss, axis=(0, 1))

    def input_grad(self, output, output_pred):
        """ Calculate input gradient given output and predicted output. """
        return self.input_grad_from_closest(output, output_pred)

    def input_grad_from_closest(self, output, output_pred):
        """
        Calculate input gradient given output and predicted output. This method works by taking the closest prototype as
        the expected value and calculating the gradient based on this. The RSLVQ implementation is also trained with the
        expected output.
         """
        # the output given here is the one-hot encoding of the expected one, which just happens to the correct class for
        # each data point given in last input
        y = self.onehot_decode(output)
        # our training data is the last input
        X = self.last_input

        assert(len(X.shape) == 2)
        assert(len(y.shape) == 1)

        if self.init:
            rv = self.compute_rslvq_gradients_from_nearest_macthing_prototype(X, y)
            #self.rslvq.partial_fit(X, y)
            self.rslvq._optimize(X, y, not self.rslvq.random_state)
        else:
            xy = unique_labels(y)
            if len(xy) != self.classAmount:
                raise ValueError('The first batch did not contain a sample of every class. Unfortunately, this is '
                                 'required for initialising the RSLVQ instance. Please rerurn the program. If this '
                                 'happens frequently, try increasing the batch size.')

            self.rslvq = nRSLVQ(sigma=.2, prototypes_per_class=4, batch_size=y.shape[0], n_epochs=4, random_state=self.rng)
            self.rslvq.fit(X, y)
            rv = self.compute_rslvq_gradients_from_nearest_macthing_prototype(X, y)
            self.init = True

        return rv * 0.006

    def input_grad_from_learning(self, output, output_pred):
        """ Calculate input gradient given output and predicted output. """
        # the output given here is the one-hot encoding of the expected one, which just happens to the correct class for
        # each data point given in last input
        y = self.onehot_decode(output)
        # our training data is the last input
        X = self.last_input

        output_grad = np.zeros(X.shape)

        for i in range(X.shape[0]):
            _X = X[i:i+1]
            _y = y[i:i+1]
            # the fit_batch method is basically the same as fit_partial, but w/o the validation which we don't need
            # because we initialized the class beforehand
            updates = -self.rslvq.fit_batch(_X, _y, 1/self.rslvq.sigma)
            output_grad[i] = np.mean(updates, 0)

        return output_grad

    def compute_rslvq_gradients_from_nearest_macthing_prototype(self, X, y):
        """
        This class computes the gradient by giving the distance to the nearest prototype with the same class
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


def run():
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
            nnet.Activation('relu'),
            nnet.Flatten(),
            RslvqLayer(n_classes, stretchToOnehot=True),
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
    run()
