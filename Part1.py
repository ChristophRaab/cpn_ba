from RSLVQ import RslvqLayer
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import ImageFile
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


if __name__ == "__main__":
    # define function to load train, test, and validation datasets
    # load train, test, and validation datasets
    train_files, train_targets = load_dataset('CodeData/dogImages/train')
    valid_files, valid_targets = load_dataset('CodeData/dogImages/valid')
    test_files, test_targets = load_dataset('CodeData/dogImages/test')

    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("CodeData/dogImages/train/*/"))]

    # load the network
    #nn = extract_xception()
    #Xception_model = Xception(weights='imagenet')
    # pre-process the data for Keras
    train_tensors = paths_to_tensor(train_files).astype('float32') / 255
    valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255
    test_tensors = paths_to_tensor(test_files).astype('float32') / 255

    # get the bottleneck features
    bottleneck_features = np.load('CodeData/DogXceptionData.npz')
    train_Xcep = bottleneck_features['train']
    valid_Xcep = bottleneck_features['valid']
    test_Xcep = bottleneck_features['test']

    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=train_Xcep.shape[1:]))
    model.add(Dense(133, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='weights.try1.hdf5',
                                   verbose=1, save_best_only=True)

    model.fit(train_Xcep, train_targets,
              validation_data=(valid_Xcep, valid_targets),
              epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

    # print statistics about the dataset
    print('There are %d total dog categories.' % len(dog_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.'% len(test_files))
