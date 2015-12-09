import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import sys
import pickle

class FullyConnectedLayer():

    def __init__(self, inputs, n_in, n_out, rng):

        self.inputs = inputs

        self.W = theano.shared(np.random.normal(size=(n_in, n_out)), borrow=True)
        self.b = theano.shared(np.random.normal(size=(n_out,)), borrow=True)
        self.params = [self.W, self.b]

        self.outputs = T.nnet.relu(T.dot(self.inputs, self.W) + self.b)

class SoftmaxLayer():

    def __init__(self, inputs, n_in, n_out, rng):

        self.inputs = inputs

        self.W = theano.shared(np.random.normal(size=(n_in, n_out)), borrow=True)
        self.b = theano.shared(np.random.normal(size=(n_out,)), borrow=True)
        self.params = [self.W, self.b]

        self.p_y_given_x = T.nnet.softmax(T.dot(self.inputs, self.W)  + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def cost(self, y):

        # y_pred_print = theano.printing.Print('y_pred value')(self.outputs)
        # y_print = theano.printing.Print('y_pred value')(y)

        return  T.nnet.categorical_crossentropy(self.p_y_given_x, y).sum()

    def errors(self, y):

        # y_pred_print = theano.printing.Print('y pred value')(self.y_pred)
        # y_print = theano.printing.Print('y target value')(y)

        # return  T.mean(T.neq(y_pred_print, y_print))
        return  T.mean(T.neq(self.y_pred, y))

class MLP():

    def __init__(self):

        self.x = T.dmatrix('x')

        rng = RandomStreams(seed=234)

        self.fc1 = FullyConnectedLayer(self.x, 9, 9, rng)
        self.softmax_layer = SoftmaxLayer(self.fc1.outputs, 9, 7, rng)

    def gradient_descent(self, cost, params, learning_rate, momentum_rate):

        updates = []

        for param in params:

            # momentum = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            # momentum_update = (momentum, momentum_rate + momentum - learning_rate * T.grad(cost, param))
            # param_udpate = (param, param + momentum)

            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            updates.append((param, param - learning_rate*param_update))
            updates.append((param_update, momentum_rate*param_update + (1. - momentum_rate)*T.grad(cost, param)))

        return updates

    # def save_error_rates(self, error_rates):
    #
    #     pickle.dump()


    def train(self,
              train_set_x, train_set_y,
              batch_size, n_epoch,
              learning_rate, momentum_rate,
              weight_decay
              ):

        self.y = T.ivector('y')
        index = T.lscalar()

        n_batch = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)

        params = self.fc1.params + self.softmax_layer.params

        # L2 = (self.fc1.W**2).sum() + (self.softmax_layer.W**2).sum()

        # cost = self.softmax_layer.cost(self.y) + weight_decay * L2
        cost = self.softmax_layer.cost(self.y)
        pred = self.softmax_layer.y_pred

        train_model = theano.function([index],
                                      [cost],
                                      updates = self.gradient_descent(cost, params, learning_rate, momentum_rate),
                                      givens = {
                                              self.x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                              self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
                                              }
                                     )

        error_model = theano.function( [index],
                                       [self.softmax_layer.errors(self.y)],
                                       givens = {
                                            self.x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                            self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
                                       }
                                     )

        error_rates = []
        for i_epoch in range(n_epoch):

            for i_batch in range(n_batch):

                train_model(i_batch)

            losses = [error_model(i) for i
                                 in range(n_batch)]

            loss = np.mean(losses)
            error_rates.append(loss)

            print('training epoch : %d - loss : %f' % (i_epoch, loss))

        return error_rates
