#from RSLVQ import RslvqLayer
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


def mapClassToIndex(arr):
    for i in range(len(arr)):
        if arr[i] > 0.99:
            return i
    return 0


def mapClassArrayToIndexArray(arr):
    return list(map(lambda x: mapClassToIndex(x), arr))


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
        return nRSLVQ(
            prototypes_per_class=self.prototypes_per_class,
            sigma=self.sigma,
            batch_size=self.batch_size,
            n_epochs=self.epochs
        )

    def __lt__(self, other):
        return self.cost < other.cost


class LvqTester:
    def __init__(self, path="CodeData", big_features=True, filter='pca', pca_dims=137):
        """
        Initializes the class (and directly starts loading data)
        :param path: The path to the data set.
        :param big_features: Whether to use big feature inputs (mean on axis 1,2 instead of 2,3)
        """
        self.mean_axis = (1, 2) if big_features else (2, 3)
        self.pca_dims = pca_dims
        self.filters = {
            "pca": self.pcaTrainingData,
            "min": self.minTrainingData,
            "max": self.maxTrainingData,
            "mean": self.meanTrainingData,
            "flatten": self.flattenTrainingDaten
        }
        self.filter = self.filters[filter]
        self.bottleneck_features = np.load(path + '/DogXceptionData.npz')

        self.train_set = self.refineTrainingData(self.bottleneck_features['train'])
        self.valid_set = self.refineTrainingData(self.bottleneck_features['valid'])
        self.test_set = self.refineTrainingData(self.bottleneck_features['test'])

        self.train_files, self.train_targets = load_dataset(path + '/dogImages/train')
        self.valid_files, self.valid_targets = load_dataset(path + '/dogImages/valid')
        self.test_files, self.test_targets = load_dataset(path + '/dogImages/test')

        self.valid_targets_indexed = self.mapTargetsToIndexed(self.valid_targets)
        self.train_targets_indexed = self.mapTargetsToIndexed(self.train_targets)
        self.test_targets_indexed = self.mapTargetsToIndexed(self.test_targets)

        self.tests = Queue(1)
        self.results = Queue(1)
        self.lock = threading.Lock()
        print("Prepared Tester: filter=%s big_features=%s pca_dims=%d" % (filter, str(big_features), pca_dims))

    def refineTrainingData(self, features):
        """
        Refines the multidimensional bottleneck features for use with RSLVQ.
        :return: The refined data
        """
        return self.filter(features)

    def meanTrainingData(self, data):
        """
        Means the multidimensional bottleneck features.
        :return: The refined data
        """
        return np.mean(data, axis=self.mean_axis)

    def maxTrainingData(self, data):
        """
        Uses max to reduce the dimensions of the bottleneck features.
        :return: The refined data
        """
        return np.max(data, axis=self.mean_axis)

    def minTrainingData(self, data):
        """
        Uses min to reduce the dimensions of the bottleneck features.
        :return: The refined data
        """
        return np.min(data, axis=self.mean_axis)

    def pcaTrainingData(self, features):
        """
        Uses a PCA to reduce the data dimensions.
        :param features: The input data.
        :param out_dims:
        :return:
        """
        # flatten all the input data for each sample
        flattened = features.reshape(features.shape[0], -1)
        # norm the input features
        flattened = (flattened - np.mean(flattened, 0)) / np.std(flattened, 0)
        pca = deco.PCA(self.pca_dims)
        pca.fit(flattened)
        return pca.transform(flattened)

    def flattenTrainingDaten(self, features):
        """
        Flattens all sample data to a single two-dimensional array without any processing.
        :param features: The input features
        """
        return features.reshape(features.shape[0], -1)

    def mapTargetsToIndexed(self, targets):
        """
        Maps NN targets (1 at the index n) to numerical targets (n)
        :param targets: The targets to map
        :return: A numpy array with mapped targets
        """
        return np.array(mapClassArrayToIndexArray(targets))

    def testLvq(self, lvq, descr=""):
        """
        Tests an LVQ implementation with the given training data and prints the results
        :param descr: A string describing the LVQ implementation.
        :param lvq: The LVQ implementation. Must have a fit and a predict method.
        :return: A tuple of (description, correct count, total count, wrong count, accurary percentage)
        """
        fitdata = lvq.fit(self.train_set, self.train_targets_indexed)

        predict = lvq.predict(self.test_set)

        diff = list(map(lambda x: 1 if x == 0 else 0, predict - self.test_targets_indexed))
        total = len(diff)
        correct = sum(diff)
        wrong = total - correct

        if descr != "":
            descr = "[%s] " % descr

        results = (descr, correct, total, wrong, (correct/total) * 100)

        print('%sCorrectly classified %d out of %d samples (%d wrong) => Accuracy of %f%%' % results)
        return results

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

    def createTestScenarios(self):
        """
        Creates a set of test parameters to check for.
        :return: A list of test parameters, sorted by cost
        """
        rslvq = nRSLVQ(prototypes_per_class=10, sigma=0.5, batch_size=200, n_epochs=20)
        tests = []
        for sigma in [0.2, 0.5, 1.0, 3.0, 5.0, 6.0]:               # 6 steps
            for batch_size in [1, 8, 16, 32, 64, 128, 256]:        # 7 steps
                for epochs in [1, 2, 4, 8, 12]:                    # 6 steps
                    for prototypes in [1, 2, 3, 4, 6, 8, 10, 12]:  # 8 steps
                        tests.append(LvqParams(sigma, prototypes, epochs, batch_size))  # = about 1,9k tests

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
                th = MProcess(target=self._threadEntry, args=(n, True,))
            else:
                th = threading.Thread(target=self._threadEntry, args=(n, True,))
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
            print('%sCorrectly classified %d out of %d samples (%d wrong) => Accuracy of %f%%' % res[i])

    def _threadEntry(self, num, isqueue):
        """
        This method is the entry point for a test thread.
        :param num: The thread index.
        """
        while True:
            lvq = None
            if isqueue:
                try:
                    lvq = self.tests.get(False)
                except QEmpty:
                    return  # ran all tests
            else:
                self.lock.acquire()
                if len(self.tests) == 0:
                    return  # ran all tests
                lvq = self.tests.pop(0)
                self.lock.release()

            print("Thread %d -- starting [%s]" % (num, str(lvq)))
            dt = self.testLvq(lvq.createRslvq(), str(lvq))

            self.lock.acquire()
            if isqueue:
                self.results.put(dt, False)
            else:
                self.results.append(dt)
            self.lock.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run RSLVQ learning accuracy tests on the Xception bottleneck feature dataset.')
    parser.add_argument('--threads', '-t', type=int, default=2, help='Number of threads to use.')
    parser.add_argument('--code-path', '-p', type=str, default='CodeData', help='Path to the learning dataset.')
    parser.add_argument('--filter', '-f', type=str, choices=['min', 'max', 'mean', 'pca', 'flatten'], default='mean',
                        help='How to prepare the bottleneck features before using them with the LVQ algorithm.')
    parser.add_argument('--big-features', action='store_true',
                        help='Whether to mean the input to big features or small ones.')
    parser.add_argument('--use-mp', '-m', action='store_true',
                        help='Whether to use the multiprocessing library instead of threading.')
    parser.add_argument('--pca-dims', type=int, default=100, help='Number of dimensions in the PCA output.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tester = LvqTester(big_features=args.big_features, path=args.code_path, filter=args.filter, pca_dims=args.pca_dims)
    # tests = tester.createTestScenarios()
    tests = [ LvqParams(sigma=.2, prototypes_per_class=8, batch_size=256, epochs=4) ]
    print('Running %d tests with%s big feature space and %d %s.'
          % (len(tests), "out" if not args.big_features else "", args.threads, "processes" if args.use_mp else "threads"))
    tester.runTestsParallel(tests, threads=args.threads, use_multiprocessing=args.use_mp)

