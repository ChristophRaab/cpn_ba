from keras import backend as K
from keras.engine.topology import Layer


class RslvqImplementation:

    def __init__(self):
        pass


# class template from keras for a layer
class RslvqLayer(Layer):
    """
    This class is a layer which applies RSVLQ classification to data
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RslvqLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(RslvqLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim
