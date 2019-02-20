from keras.preprocessing import image
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def path_to_tensor(img_path):
    """
    This method loads an image and returns it as tensor with the shape (1, 224, 224, 3).
    :param img_path: The image file.
    :return: A tensor in the shape (1, 224, 224, 3).
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    """
    This method loads images and returns them as tensor with the shape (#, 224, 224, 3).
    :param img_paths: A list of image paths.
    :return: A tensor in the shape (#, 224, 224, 3).
    """
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def extract_xception(tensor):
    """
    This method loads Xception, pre-trained with the ImageNet data set, and runs the given input against it. After that
    it returns the bottleneck features.
    :param tensor: The input values for Xception.
    :return: The bottleneck features for the given input.
    """
    from keras.applications.xception import Xception, preprocess_input
    return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def load_dataset(path, onehot=True):
    """
    This method loads an image data set from a folder structure like this:
    folder
    |- class01
    ||- image01.jpg
    ||- image02.jpg
    ||- ...
    |- class02
    ||- ...
    |- ...
    It returns a tuple with the file names and the corresponding category.
    :param path: The path to the folder.
    :param onehot: Whether to return the category as data or as onehot-encoded array.
    :return: A tuple of (files, category).
    """
    data = load_files(path)
    dog_files = np.array(data['filenames'])

    dog_targets = np.array(data['target'])
    if onehot:
        dog_targets = np_utils.to_categorical(dog_targets, 133)

    return dog_files, dog_targets
