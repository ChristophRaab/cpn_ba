import time
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from glob import glob
from common import load_dataset, paths_to_tensor


if __name__ == "__main__":
    # define function to load train, test, and validation datasets
    # load train, test, and validation datasets
    train_files, train_targets = load_dataset('CodeData/dogImages/train')
    valid_files, valid_targets = load_dataset('CodeData/dogImages/valid')
    test_files, test_targets = load_dataset('CodeData/dogImages/test')

    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("CodeData/dogImages/train/*/"))]

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
    names = model.metrics_names

    # uncomment if the weights are needed
    # checkpointer = ModelCheckpoint(filepath='weights.try1.hdf5', verbose=1, save_best_only=True)

    model.fit(train_Xcep, train_targets, validation_data=(valid_Xcep, valid_targets), epochs=20, batch_size=20, verbose=1)
    eval_result = model.evaluate(test_Xcep, test_targets, batch_size=20, verbose=1)

    print("Evaluated: %s = %f ,  %s = %f\n" % (model.metrics_names[0], eval_result[0], model.metrics_names[1], eval_result[1]))

    starttime = time.time()
    test_predict = model.predict(test_Xcep, batch_size=20)
    endditime = time.time()
    print('prediction took %0.3f ms' % ((endditime - starttime) * 1000.0))
