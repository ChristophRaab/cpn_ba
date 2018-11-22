#from RSLVQ import RslvqLayer
from sklearn_lvq import RslvqModel
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import keras.backend as kbackend
from keras.callbacks import ModelCheckpoint
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from RSLVQ.rslvq import RSLVQ as nRSLVQ
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
    def __init__(self, path="CodeData", big_features=False):
        """
        Initializes the class (and directly starts loading data)
        :param path: The path to the data set.
        :param big_features: Whether to use big feature inputs (mean on axis 1,2 instead of 2,3)
        """
        self.mean_axis = (1, 2) if big_features else (2, 3)
        self.bottleneck_features = np.load(path + '/DogXceptionData.npz')

        self.train_set = self.meanBottleneckFeatures(self.bottleneck_features['train'])
        self.valid_set = self.meanBottleneckFeatures(self.bottleneck_features['valid'])
        self.test_set = self.meanBottleneckFeatures(self.bottleneck_features['test'])

        self.train_files, self.train_targets = load_dataset(path + '/dogImages/train')
        self.valid_files, self.valid_targets = load_dataset(path + '/dogImages/valid')
        self.test_files, self.test_targets = load_dataset(path + '/dogImages/test')

        self.valid_targets_indexed = self.mapTargetsToIndexed(self.valid_targets)
        self.train_targets_indexed = self.mapTargetsToIndexed(self.train_targets)
        self.test_targets_indexed = self.mapTargetsToIndexed(self.test_targets)

        self.tests = []
        self.results = []
        self.lock = threading.Lock()

    def meanBottleneckFeatures(self, features):
        """
        Flattens the multidimensional bottleneck featues
        :return:
        """
        return np.mean(features, axis=self.mean_axis)

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
        for sigma in [0.2, 0.4, 0.6, 0.8, 1.0]:                    # 5 steps
            for batch_size in [1, 5, 10, 20, 50, 100, 200, 400]:   # 8 steps
                for epochs in [1, 2, 4, 8, 12, 16]:                # 6 steps
                    for prototypes in [1, 2, 3, 4, 6, 8, 10, 12]:  # 8 steps
                        tests.append(LvqParams(sigma, prototypes, epochs, batch_size))  # = about 1,9k tests

        return sorted(tests)

    def runTestsParallel(self, tests, threads=3):
        """
        Starts parallel test threads. For now, this method is _not_ threadsafe!
        :param tests: The tests to run.
        :param threads: The amount of threads to use
        :return:
        """
        if len(self.tests) != 0:
            raise PermissionError("This method is still running in another thread!")

        self.tests = tests
        self.results = []
        thls = []  # actually threads, but that name was already taken

        for n in range(threads):
            th = threading.Thread(target=self._threadEntry, args=(n,))
            th.start()
            thls.append(th)

        for n in range(threads):
            thls[n].join()

        self.results.sort(key=lambda x: x[4])  # sort by accuracy
        print("")
        print(" Results ")
        print("---------")
        for i in range(len(self.results)):
            print('%sCorrectly classified %d out of %d samples (%d wrong) => Accuracy of %f%%' % self.results[i])

    def _threadEntry(self, num):
        """
        This method is the entry point for a test thread.
        :param num: The thread index.
        """
        while True:
            self.lock.acquire()
            if len(self.tests) == 0:
                return  # ran all tests
            lvq = self.tests.pop(0)
            self.lock.release()

            print("Thread %d -- starting [%s]" % (num, str(lvq)))
            dt = self.testLvq(lvq.createRslvq(), str(lvq))

            self.lock.acquire()
            self.results.append(dt)
            self.lock.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run RSLVQ learning accuracy tests on the Xception bottleneck feature dataset.')
    parser.add_argument('--threads', '-t', type=int, default=2, help='Number of threads to use.')
    parser.add_argument('--code-path', '-f', type=str, default='CodeData', help='Path to the learning dataset.')
    parser.add_argument('--big-features', action='store_true',
                        help='Whether to mean the input to big features or small ones.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tester = LvqTester(big_features=args.big_features, path=args.code_path)
    tests = tester.createTestScenarios()
    print('Running %d tests with%s big feature space and %d threads.'
          % (len(tests), "out" if not args.big_features else "", args.threads))
    tester.runTestsParallel(tests, threads=args.threads)

