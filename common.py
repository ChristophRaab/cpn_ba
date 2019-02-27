from keras.preprocessing import image
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from scipy import io
import scipy as sp
from sklearn.utils import Bunch
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numbers


def path_to_tensor(img_path, data_format=None):
    """
    This method loads an image and returns it as tensor with the shape (1, 224, 224, 3) or (1, 3, 224, 244).
    :param img_path: The image file.
    :param data_format: Allows to change the data to channels first.
    :return: A tensor in the shape (1, 224, 224, 3) or (1, 3, 224, 244).
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img, data_format=data_format)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths, data_format=None):
    """
    This method loads images and returns them as tensor with the shape (#, 224, 224, 3) or (#, 3, 224, 224).
    :param img_paths: A list of image paths.
    :param data_format: Allows to change the data to channels first.
    :return: A tensor in the shape (#, 224, 224, 3) or (#, 3, 244, 244).
    """
    list_of_tensors = [path_to_tensor(img_path, data_format=data_format) for img_path in tqdm(img_paths)]
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


def fetch_mldata(dataname, target_name='label', data_name='data', transpose_data=True, data_home=None):
    """Fetch an mldata.org data set

    mldata.org is no longer operational.
    NOTE: This is a stubbed version which can only load mnist from a local file!

    If the file does not exist yet, it is downloaded from mldata.org .

    mldata.org does not have an enforced convention for storing data or
    naming the columns in a data set. The default behavior of this function
    works well with the most common cases:

      1) data values are stored in the column 'data', and target values in the
         column 'label'
      2) alternatively, the first column stores target values, and the second
         data values
      3) the data array is stored as `n_features x n_samples` , and thus needs
         to be transposed to match the `sklearn` standard

    Keyword arguments allow to adapt these defaults to specific data sets
    (see parameters `target_name`, `data_name`, `transpose_data`, and
    the examples below).

    mldata.org data sets may have multiple columns, which are stored in the
    Bunch object with their original name.

    Parameters
    ----------

    dataname : str
        Name of the data set on mldata.org,
        e.g.: "leukemia", "Whistler Daily Snowfall", etc.
        The raw name is automatically converted to a mldata.org URL .

    target_name : optional, default: 'label'
        Name or index of the column containing the target values.

    data_name : optional, default: 'data'
        Name or index of the column containing the data.

    transpose_data : optional, default: True
        If True, transpose the downloaded data array.

    data_home : optional, default: None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    Returns
    -------

    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'DESCR', the full description of the dataset, and
        'COL_NAMES', the original names of the dataset columns.
    """

    # normalize dataset name
    filename = 'mnist.mat'

    # load dataset matlab file
    with open(filename, 'rb') as matlab_file:
        matlab_dict = io.loadmat(matlab_file, struct_as_record=True)

    # -- extract data from matlab_dict

    # flatten column names
    col_names = [str(descr[0])
                 for descr in matlab_dict['mldata_descr_ordering'][0]]

    # if target or data names are indices, transform then into names
    if isinstance(target_name, numbers.Integral):
        target_name = col_names[target_name]
    if isinstance(data_name, numbers.Integral):
        data_name = col_names[data_name]

    # rules for making sense of the mldata.org data format
    # (earlier ones have priority):
    # 1) there is only one array => it is "data"
    # 2) there are multiple arrays
    #    a) copy all columns in the bunch, using their column name
    #    b) if there is a column called `target_name`, set "target" to it,
    #        otherwise set "target" to first column
    #    c) if there is a column called `data_name`, set "data" to it,
    #        otherwise set "data" to second column

    dataset = {'DESCR': 'mldata.org dataset: %s' % dataname,
               'COL_NAMES': col_names}

    # 1) there is only one array => it is considered data
    if len(col_names) == 1:
        data_name = col_names[0]
        dataset['data'] = matlab_dict[data_name]
    # 2) there are multiple arrays
    else:
        for name in col_names:
            dataset[name] = matlab_dict[name]

        if target_name in col_names:
            del dataset[target_name]
            dataset['target'] = matlab_dict[target_name]
        else:
            del dataset[col_names[0]]
            dataset['target'] = matlab_dict[col_names[0]]

        if data_name in col_names:
            del dataset[data_name]
            dataset['data'] = matlab_dict[data_name]
        else:
            del dataset[col_names[1]]
            dataset['data'] = matlab_dict[col_names[1]]

    # set axes to scikit-learn conventions
    if transpose_data:
        dataset['data'] = dataset['data'].T
    if 'target' in dataset:
        if not sp.sparse.issparse(dataset['target']):
            dataset['target'] = dataset['target'].squeeze()

    return Bunch(**dataset)

