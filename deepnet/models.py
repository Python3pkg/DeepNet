import numpy as np
import theano
import theano.tensor as T

from .layers import FullyConnectedLayer, SoftmaxLayer, PoolLayer, ConvLayer
from .net import Network

class ConvNet(Network):

    def __init__(self, train_set_x, train_set_y, valid_set_x, valid_set_y, batch_size, nkerns=[20, 50, 50]):

        super().__init__(train_set_x, train_set_y, valid_set_x, valid_set_y, batch_size)

        self.x = T.dtensor4('x')

        nb_channel = train_set_x.shape[1]
        height = train_set_x.shape[2]
        width = train_set_x.shape[3]

        self.layer0 = ConvLayer(
            self.rng,
            inputs=self.x,
            image_shape=(batch_size, nb_channel, height, width),
            filter_shape=(nkerns[0], nb_channel, 6, 6),
         #    stride=2,
         #    pad=2
        )

        self.layer1 = PoolLayer(
            inputs=self.layer0.output,
            input_shape=self.layer0.output_shape
        )

        self.layer2 = ConvLayer(
            self.rng,
            inputs=self.layer1.output,
            image_shape=self.layer1.output_shape,
            filter_shape=(nkerns[1], nkerns[0], 6, 6),
         #    stride=2,
         #    pad=2
        )

        self.layer3 = PoolLayer(
            inputs=self.layer2.output,
            input_shape=self.layer2.output_shape
        )

        layer4_input = self.layer3.output.flatten(2)

        n_in = self.layer3.output_shape[1] * self.layer3.output_shape[2] * self.layer3.output_shape[3]
        n_out = int(n_in/2)

        self.layer4 = FullyConnectedLayer(layer4_input, n_in, n_out, self.rng)

        nb_outputs = len(np.unique(train_set_y))
        self.layer5 = SoftmaxLayer(inputs=self.layer4.outputs, n_in=n_out, n_out=nb_outputs, rng=self.rng)

        self.layers = [self.layer0, self.layer2, self.layer4, self.layer5]

class MLP(Network):

    def __init__(self, train_set_x, train_set_y, valid_set_x, valid_set_y, batch_size):

        super().__init__(train_set_x, train_set_y, valid_set_x, valid_set_y, batch_size)

        self.x = T.dmatrix('x')

        nb_features = self.train_set_x.get_value().shape[1]
        nb_outputs = len(np.unique(self.train_set_y.get_value()))

        self.fc1 = FullyConnectedLayer(self.x, nb_features, nb_features, self.rng)
        self.softmax_layer = SoftmaxLayer(self.fc1.outputs, nb_features, nb_outputs, self.rng)

        self.layers = [self.fc1, self.softmax_layer]
