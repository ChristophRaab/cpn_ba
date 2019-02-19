import time

from sklearn_lvq import RslvqModel
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import keras.backend as kbackend
import sklearn.decomposition as deco
from keras.callbacks import ModelCheckpoint
from queue import Empty as QEmpty
from sklearn.datasets import load_files
from sklearn.utils import shuffle as combined_shuffle
from keras.utils import np_utils
import numpy as np
from RSLVQ.rslvq import RSLVQ as nRSLVQ
from multiprocessing import Process as MProcess, Queue, Lock as MLock
from glob import glob
from tqdm import tqdm
import pickle
from PIL import ImageFile
import threading
import argparse
ImageFile.LOAD_TRUNCATED_IMAGES = True


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def extract_xception(tensor):
    from keras.applications.xception import Xception, preprocess_input
    return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


class LvqParams:
    def __init__(self, sigma, prototypes_per_class, epochs, batch_size):
        self.sigma = sigma
        self.prototypes_per_class = prototypes_per_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.cost = self.computeCost()

    def computeCost(self):
        """
        Computes a loose approximation of the runtime cost
        :return:
        """
        # this assumes that a higher batch size can process more data at once and is therefore faster
        return self.prototypes_per_class * self.prototypes_per_class * self.epochs * (1/self.batch_size)

    def __str__(self):
        """
        Returns a string represenation of the class
        :return:
        """
        return "LVQ with sigma=%f and %d prototypes per class; using batch size %d on %d epochs" % (
            self.sigma, self.prototypes_per_class, self.batch_size, self.epochs
        )

    def createRslvq(self):
        """
        Creates an instance of the RSLVQ implementation using the given parameters
        :return:
        """
        return LvqClassifierLayer(
            prototypes_per_class=self.prototypes_per_class,
            sigma=self.sigma,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )

    def test_lvq(self, train_set, train_targets, test_set, test_targets, description=""):
        """
        Tests an LVQ implementation with the given training and validation data.
        :param test_targets: The actual classification for the test data.
        :param test_set: The test samples.
        :param train_targets: The correct classification for the training samples.
        :param train_set: The training samples.
        :param description: A string describing the LVQ implementation.
        :return:
        A tuple of (description, correct count, total count, wrong count, accurary percentage, prediction time in ms)
        """
        lvq = self.createRslvq()
        fitdata = lvq.fit(train_set, train_targets)

        starttime = time.time()
        predict = lvq.predict(test_set)
        endtime = time.time()
        prediction_time_ms = (endtime - starttime) * 1000.0

        diff = list(map(lambda x: 1 if x == 0 else 0, predict - test_targets))
        total = len(diff)
        correct = sum(diff)
        wrong = total - correct

        if description != "":
            description = "[%s] " % description

        results = (description, correct, total, wrong, (correct / total) * 100, prediction_time_ms)

        return results

    def __lt__(self, other):
        return self.cost < other.cost


class DimensionReduction:
    """
    This class provides dimensional reduction for input data.
    """

    def __init__(self, filter='flatten', mean_dimensions=None, pca_dims=32):
        """
        Initializes the dimension reduction.
        :param filter:
        The function to use for reducing the data dimensions to two. Can be 'min', 'max', 'mean', 'pca' or 'flatten'.
        :param mean_dimensions:
        The dimensions to reduce. For example, if one has a 4-dimensional array and the given value is (1,2) the new
        array will have the shape [old.shape[0],old.shape[3]]. Ignored for filter='flatten'. Defaults to old.shape[2:].
        :param pca_dims: The dimensions to use when using a pca.
        """
        self.mean_dimensions = mean_dimensions
        self.pca_dims = pca_dims
        self.filters = {
            "pca": self._pca_data,
            "min": self._min_data,
            "max": self._max_data,
            "mean": self._mean_data,
            "flatten": self._flatten_data
        }

        if filter not in self.filters:
            raise ValueError("Unknown filter function '%s'!" % filter)
        self.filter = self.filters[filter]

    def refine(self, data):
        """
        Reduces the multi-dimensional input data to a two-dimensional array with the given filter.
        :return: The refined data
        """
        shapelen = len(data.shape)

        if shapelen < 2:
            raise ValueError("Input data has too few dimensions!")
        elif shapelen == 2:
            return data

        mean_dim = self.mean_dimensions
        if mean_dim is None:
            # no shape given; simply remove all extra dimensions
            mean_dim = data.shape[2:]
        elif len(mean_dim) != shapelen - 2:
            raise ValueError("The mean dimensions do not match the input!")

        return self.filter(data, mean_dim)

    def _mean_data(self, data, mean_axis):
        """
        Means the multidimensional data.
        :return: The refined data
        """
        return np.mean(data, axis=mean_axis)

    def _max_data(self, data, mean_axis):
        """
        Uses max to reduce the dimensions of the data.
        :return: The refined data
        """
        return np.max(data, axis=mean_axis)

    def _min_data(self, data, mean_axis):
        """
        Uses min to reduce the dimensions of the data.
        :return: The refined data
        """
        return np.min(data, axis=mean_axis)

    def _pca_data(self, flattened):
        """
        Uses a PCA to reduce the data dimensions.
        :param flattened: The input data.
        :return:
        """
        # flatten all the input data for each sample
        dim1size = flattened.shape[0]
        flattened = flattened.reshape(dim1size, -1)
        # norm the input features
        pca = deco.PCA(self.pca_dims)
        pca.fit(flattened[0:200])
        return pca.transform(flattened)

    def _flatten_data(self, features, mean_axis=None):
        """
        Flattens all sample data to a single two-dimensional array without any processing.
        :param features: The input features
        """
        return features.reshape(features.shape[0], -1)


class LvqTester:
    def __init__(self, path="CodeData", big_features=True, filter='flatten', pca_dims=137, shuffle=False):
        """
        Initializes the class (and directly starts loading data)
        :param path: The path to the data set.
        :param big_features: Whether to use big feature inputs (mean on axis 1,2 instead of 2,3)
        """
        self.mean_axis = (1, 2) if big_features else (2, 3)
        self.filter = DimensionReduction(filter, self.mean_axis, pca_dims)

        self.bottleneck_features = np.load(path + '/DogXceptionData.npz')

        self.train_set = self.filter.refine(self.bottleneck_features['train'])
        self.valid_set = self.filter.refine(self.bottleneck_features['valid'])
        self.test_set = self.filter.refine(self.bottleneck_features['test'])

        self.train_files, self.train_targets = load_dataset(path + '/dogImages/train')
        self.valid_files, self.valid_targets = load_dataset(path + '/dogImages/valid')
        self.test_files, self.test_targets = load_dataset(path + '/dogImages/test')

        self.valid_targets_indexed = self.mapTargetsToIndexed(self.valid_targets)
        self.train_targets_indexed = self.mapTargetsToIndexed(self.train_targets)
        self.test_targets_indexed = self.mapTargetsToIndexed(self.test_targets)

        # this part possibly shuffles, but always merges train and valid.
        if shuffle:
            combined_set = np.concatenate((self.train_set, self.valid_set, self.test_set), axis=0)
            combined_targets = np.concatenate((self.train_targets_indexed, self.valid_targets_indexed, self.test_targets_indexed), axis=0)

            combined_set_shuffled, combined_targets_shuffled = combined_shuffle(combined_set, combined_targets)
            self.train_set = combined_set_shuffled[self.test_set.shape[0]:]
            self.train_targets_indexed = combined_targets_shuffled[self.test_set.shape[0]:]
            self.test_set = combined_set_shuffled[:self.test_set.shape[0]:]
            self.test_targets_indexed = combined_targets_shuffled[:self.test_set.shape[0]]
        else:
            self.train_set = np.concatenate((self.train_set, self.valid_set), axis=0)
            self.train_targets = np.concatenate((self.train_targets, self.valid_targets), axis=0)

        self.tests = Queue(1)
        self.results = Queue(1)
        self.lock = threading.Lock()
        print("Prepared Tester: filter=%s big_features=%s pca_dims=%d" % (filter, str(big_features), pca_dims))

    def mapTargetsToIndexed(self, targets):
        """
        Maps neural network targets (1 at the index n) to numerical targets (n).
        :param targets: The targets to map
        :return: A numpy array with mapped targets
        """
        def mapClassToIndex(arr):
            for i in range(len(arr)):
                if arr[i] > 0.99:
                    return i
            raise ValueError('No class found in array!')

        # return np.array(list(map(lambda x: mapClassToIndex(x), targets)))
        return np.array([mapClassToIndex(sample) for sample in targets])

    def frange(self, x, y, jump):
        """
        Like range(), but for floats
        :param x: Start of the range
        :param y: (inclusive) End of the range
        :param jump: Jump size
        :return:
        """
        while x <= y:
            yield x
            x += jump

    def createTestScenarios(self, mean_dims=None):
        """
        Creates a set of test parameters to check for.
        :return: A list of test parameters, sorted by cost
        """
        tests = []
        for sigma in [0.2, 0.5, 1.0, 3.0, 5.0, 6.0]:               # 6 steps
            for batch_size in [1, 8, 16, 32, 64, 128, 256]:        # 7 steps
                for epochs in [1, 2, 4, 8, 12]:                    # 6 steps
                    for prototypes in [1, 2, 3, 4, 6, 8, 10, 12]:  # 8 steps
                        # about 1,9k tests
                        tests.append(LvqParams(sigma, prototypes, epochs, batch_size, mean_dimensions=mean_dims))

        return sorted(tests)

    def runTestsParallel(self, tests, threads=3, use_multiprocessing=False):
        """
        Starts parallel test threads. For now, this method is _not_ threadsafe!
        :param tests: The tests to run.
        :param threads: The amount of threads to use
        :param use_multiprocessing: Whether to use the multiprocessing library instead of thread
        :return:
        """
        if not self.tests.empty():
            raise PermissionError("This method is still running in another thread!")

        self.tests = Queue(len(tests))
        self.results = Queue(len(tests))
        for i in range(len(tests)):
            self.tests.put(tests[i], False)

        thls = []  # actually threads, but that name was already taken

        for n in range(threads):
            th = None
            if use_multiprocessing:
                th = MProcess(target=self._threadEntry, args=(n,))
            else:
                th = threading.Thread(target=self._threadEntry, args=(n,))
            th.start()
            thls.append(th)

        for n in range(threads):
            thls[n].join()

        res = []
        while True:  # read everything from the query
            try:
                res.append(self.results.get_nowait())
            except QEmpty:
                break

        res.sort(key=lambda x: x[4])  # sort by accuracy
        print("")
        print(" Results ")
        print("---------")
        for i in range(len(res)):
            print('%sCorrectly classified %d out of %d samples (%d wrong) => Accuracy of %f%% (in %f ms)' % res[i])

    def _threadEntry(self, num):
        """
        This method is the entry point for a test thread.
        :param num: The thread index.
        """
        while True:
            lvq = None
            try:
                lvq = self.tests.get(True)
            except QEmpty:
                return  # ran all tests

            print("Thread %d -- starting [%s]" % (num, str(lvq)))
            dt = lvq.test_lvq(
                self.train_set, self.train_targets_indexed, self.test_set, self.test_targets_indexed, str(lvq))
            print('%sCorrectly classified %d out of %d samples (%d wrong) => Accuracy of %f%% (%f ms)' % dt)

            self.lock.acquire()
            self.results.put(dt, False)
            self.lock.release()


class LvqClassifierLayer:
    def __init__(self, sigma=0.2, prototypes_per_class=8, batch_size=256, epochs=4):
        """
        Initializes the RSLVQ training layer.
        :param sigma: The sigma to use for the RSLVQ classifier.
        :param prototypes_per_class: The number of prototypes used for each class in the RSLVQ classifier.
        :param batch_size: The batch size used in the RSLVQ classifier.
        :param epochs: The number of training epochs.
        """
        self.sigma = sigma
        self.prototypes_per_class = prototypes_per_class
        self.batch_size = batch_size
        self.epochs = epochs

        self.lvq = nRSLVQ(
            prototypes_per_class=self.prototypes_per_class,
            sigma=self.sigma,
            batch_size=self.batch_size,
            n_epochs=self.epochs
        )

    def fit(self, data, classification):
        """
        Trains the classifier on the given data.
        :param data:
        The data to train on. This must be an at least two-dimensional array where each element of the first dimension
        represents the (possibly multidimensional) data for a sample.
        :param classification:
        The classification for each sample of the first array. This must be an one-dimensional array where is element is
        an int representing the class of the sample at the same index in the data array. The length of this array must
        match the length of the data array.
        """
        if len(data) != len(classification):
            raise ValueError('The provided classification data does not match the training data!')

        if len(data.shape) < 2:
            raise ValueError('The data array has too few dimensions!')

        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)

        self.lvq.fit(data, classification)

    def predict(self, data):
        """
        Predicts the classes for the given input data.
        :param data: The data to analyze.
        :return: An one-dimensional array where each entry contains the predicted class for the input sample.
        """
        if len(data.shape) < 2:
            raise ValueError('The data array has too few dimensions!')

        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)

        return self.lvq.predict(data)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run RSLVQ learning accuracy tests on the Xception bottleneck feature dataset.')
    parser.add_argument('--threads', '-t', type=int, default=2, help='Number of threads to use.')
    parser.add_argument('--code-path', '-p', type=str, default='CodeData', help='Path to the learning dataset.')
    parser.add_argument('--filter', '-f', type=str, choices=['min', 'max', 'mean', 'pca', 'flatten'], default='flatten',
                        help='How to prepare the bottleneck features before using them with the LVQ algorithm.')
    parser.add_argument('--big-features', action='store_true',
                        help='Whether to mean the input to big features or small ones.')
    parser.add_argument('--use-mp', '-m', action='store_true',
                        help='Whether to use the multiprocessing library instead of threading.')
    parser.add_argument('--shuffle', '-s', action='store_true', help='Whether to shuffle the input data set.')
    parser.add_argument('--pca-dims', type=int, default=100, help='Number of dimensions in the PCA output.')
    return parser.parse_args()


def best_tests():
    # the parameters which returned the best results in long-running tests
    return [
        LvqParams(sigma=.2, prototypes_per_class=8, batch_size=256, epochs=4),
        LvqParams(sigma=6, prototypes_per_class=12, batch_size=128, epochs=4),
        LvqParams(sigma=6, prototypes_per_class=12, batch_size=16, epochs=12),
        LvqParams(sigma=1, prototypes_per_class=3, batch_size=8, epochs=4),
        LvqParams(sigma=5, prototypes_per_class=10, batch_size=32, epochs=8),
        LvqParams(sigma=.2, prototypes_per_class=10, batch_size=32, epochs=8),
        LvqParams(sigma=6, prototypes_per_class=10, batch_size=128, epochs=2),
        LvqParams(sigma=3, prototypes_per_class=10, batch_size=8, epochs=4),
        LvqParams(sigma=1, prototypes_per_class=12, batch_size=128, epochs=12),
        LvqParams(sigma=.2, prototypes_per_class=8, batch_size=256, epochs=1)
    ]


if __name__ == "__main__":
    args = parse_args()
    tester = LvqTester(big_features=args.big_features, path=args.code_path, filter=args.filter, pca_dims=args.pca_dims, shuffle=args.shuffle)
    tests = best_tests()
    print('Running %d tests with%s big feature space%s and %d %s.'
          % (len(tests),
             "out" if not args.big_features else "",
             " (shuffled)" if args.shuffle else "",
             args.threads,
             "processes" if args.use_mp else "threads"))
    tester.runTestsParallel(tests, threads=args.threads, use_multiprocessing=args.use_mp)

