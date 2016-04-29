import os
import sys
import time
import argparse

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet import conv

import lasagne
from lasagne import layers, nonlinearities
from lasagne.theano_extensions import padding


from data import load_cifar10, load_cifar100
from utils import tile_raster_images
try:
    from PIL import Image
except:
    import Image

class DoubleConvLayer(layers.conv.BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1,1),
                 pad=0, untie_biases=False, kernel_size=3, kernel_pool_size=1,
                 W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
                 convolution=theano.tensor.nnet.conv2d, **kwargs):
        super(DoubleConvLayer, self).__init__(incoming, num_filters, filter_size,
                                              stride, 0, untie_biases, W, b,
                                              nonlinearity, flip_filters, n=2,
                                              **kwargs)
        self.convolution = convolution
        self.kernel_size = kernel_size
        self.pool_size = kernel_pool_size
        self.filter_offset = self.filter_size[0] - self.kernel_size + 1

        self.n_times = self.filter_offset ** 2
        self.rng = RandomStreams(123)

    def convolve(self, input, **kwargs):
        
        border_mode = 'half'

        W_shape = (self.num_filters * self.n_times,  self.input_shape[1]) + \
                  (self.kernel_size,)*2

        filter = T.reshape(T.eye(numpy.prod(W_shape[1:])),
                           (numpy.prod(W_shape[1:]),) + W_shape[1:])
        
        W_effective = self.convolution(self.W, filter,
                                       border_mode='valid',
                                       filter_flip=False)
        
        W_effective = T.reshape(W_effective.dimshuffle(0, 2, 3, 1), W_shape)
        
        output = self.convolution(input, W_effective,
                                  self.input_shape, W_shape,
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        
        if self.pool_size == -1:
            this_shape = (input.shape[0], self.num_filters, self.n_times) + self.input_shape[2:]
            output = T.reshape(output, this_shape).max(axis=2)
            
        elif self.pool_size != 1:
            this_shape = (input.shape[0] * self.num_filters,) + (self.filter_offset,)*2 \
                         + (numpy.prod(self.input_shape[2:]),)
            output = T.reshape(output, this_shape).dimshuffle(0, 3, 1, 2)
            output = T.signal.pool.pool_2d(output, (self.pool_size,)*2, ignore_border=True)
            this_shape = (input.shape[0], -1) + self.input_shape[2:]
            output = T.reshape(output.dimshuffle(0, 2, 3, 1), this_shape)

        return output

    def get_output_shape_for(self, input_shape):
        if self.pool_size == -1:
            n = 1
        else:
            n = self.n_times / (self.pool_size**2)
        return (input_shape[0], self.num_filters * n) + input_shape[2:]
        
class Model:
    def __init__(
        self,
        image_shape,
        filter_shape,
        num_class,
        doubleconv,
        kernel_size,
        kernel_pool_size,
    ):
        """

        """
        self.filter_shape = filter_shape
        self.n_visible = numpy.prod(image_shape)
        self.n_layers = len(filter_shape)
        self.rng = RandomStreams(123)
        self.x = T.matrix()
        self.y = T.ivector()

        NoiseLayer = layers.DropoutLayer

        self.l_input = layers.InputLayer((None, self.n_visible), self.x)
        this_layer = layers.ReshapeLayer(self.l_input, ([0],) + image_shape)

        for l in range(self.n_layers):
            activation = lasagne.nonlinearities.rectify
            convlayer = DoubleConvLayer if doubleconv else layers.Conv2DLayer
            if len(filter_shape[l]) == 3:
                if doubleconv:
                    this_layer = DoubleConvLayer(this_layer,
                                                 filter_shape[l][0],
                                                 filter_shape[l][1:],
                                                 pad='same',
                                                 nonlinearity=activation,
                                                 kernel_size=kernel_size,
                                                 kernel_pool_size=kernel_pool_size)
                else:
                    this_layer = layers.Conv2DLayer(this_layer,
                                                    filter_shape[l][0],
                                                    filter_shape[l][1:],
                                                    pad='same',
                                                    nonlinearity=activation)
                this_layer = layers.batch_norm(this_layer)

            elif len(filter_shape[l]) == 2:
                this_layer = layers.MaxPool2DLayer(this_layer, filter_shape[l])
                this_layer = NoiseLayer(this_layer, 0.5)
            elif len(filter_shape[l]) == 1:
                raise NotImplementedError

        self.top_conv_layer = this_layer
        this_layer = layers.GlobalPoolLayer(this_layer, T.mean)
        self.clf_layer = layers.DenseLayer(this_layer,
                                           num_class,
                                           W=lasagne.init.Constant(0.),
                                           nonlinearity=T.nnet.softmax
        )
        
        self.params = layers.get_all_params(self.clf_layer, trainable=True)

        self.params_all = layers.get_all_params(self.clf_layer)

    def build_model(self):
        print 'building model ...'
        fn = {}
        lr = T.scalar()
        
        pred = layers.get_output(self.clf_layer, deterministic=False)            
        cost = lasagne.objectives.categorical_crossentropy(pred, self.y)
        updates = lasagne.updates.adadelta(T.mean(cost), self.params, learning_rate=lr)
        pred_static = layers.get_output(self.clf_layer, deterministic=True)            
        error = 1. - lasagne.objectives.categorical_accuracy(pred_static, self.y)
        fn['train'] = theano.function([self.x, self.y, lr], [error, cost], updates=updates)
        fn['test'] = theano.function([self.x, self.y], error)
        fn['updates'] = theano.function([self.x, self.y], T.grad(T.mean(cost), self.params))

        return fn

    def save_model(self, saveto='model_saved.npz'):
        pp = [tp.get_value() for tp in self.params_all]
        numpy.savez(saveto, pp=pp)

    def load_model(self, loadfrom='model_saved.npz'):
        pp = numpy.load(loadfrom)['pp']
        for tp, p in zip(self.params_all, pp):
            tp.set_value(p)

            
def save_images(X, file_name, image_shape=(28, 28), tile_shape=(10, 10), color=False):
    if color:
        img_size = numpy.prod(image_shape)
        X = (X[:, :img_size], X[:, img_size:2*img_size], X[:, 2*img_size:], None)
    image = Image.fromarray(
        tile_raster_images(X=X,
                           img_shape=image_shape,
                           tile_shape=tile_shape,
                           tile_spacing=(1, 1))
    )
    image.save(file_name)


def Shape(s):
    return tuple(map(int, s.split(',')))

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='cifar10')
parser.add_argument('-train_epochs', type=int, default=100)
parser.add_argument('-patience', type=int, default=10)
parser.add_argument('-lr', type=numpy.float32, default=1e0)
parser.add_argument('-filter_shape', type=Shape, nargs='+', default=[(64, 3, 3), (2,2)])
parser.add_argument('-kernel_size', type=int, default=3)
parser.add_argument('-kernel_pool_size', type=int, default=-1)
parser.add_argument('-batch_size', type=int, default=200)
parser.add_argument('-load_model', type=int, default=0)
parser.add_argument('-save_model', type=str, default='model_saved.npz')
parser.add_argument('-train_on_valid', type=int, default=1)
parser.add_argument('-doubleconv', type=int, default=0)

if __name__ == '__main__':
    args = parser.parse_args()
    opt = vars(args)
    dataset = opt['dataset']
    train_epochs = opt['train_epochs']
    patience = opt['patience']
    lr = opt['lr']
    filter_shape = opt['filter_shape']
    kernel_size = opt['kernel_size']
    kernel_pool_size = opt['kernel_pool_size']
    batch_size = opt['batch_size']
    load_model = opt['load_model']
    save_model = opt['save_model']
    train_on_valid = opt['train_on_valid']
    doubleconv = opt['doubleconv']

    if save_model is not 'none':
        saveto = './saved/' + dataset + '_' + save_model
    else:
        saveto = None

    if load_model:
        loadfrom = saveto
    else:
        loadfrom = None

    if dataset == 'cifar10':
        load_data = load_cifar10.load_data
        image_shape = (3, 32, 32)
        color = True
        datasets = load_data()
        train_x, valid_x, test_x = [data[0] for data in datasets]
        train_y, valid_y, test_y = [data[1] for data in datasets]

        unlabeled = load_cifar100.load_data()
        unlabeled_x = numpy.concatenate([data[0] for data in unlabeled], axis=0)
        num_unlabeled = unlabeled_x.shape[0]
        
        num_train, num_valid, num_test = [data[0].shape[0] for data in datasets]
        num_class = numpy.max(train_y) + 1

    if dataset == 'cifar100':
        load_data = load_cifar100.load_data
        image_shape = (3, 32, 32)
        color = True
        datasets = load_data()
        train_x, valid_x, test_x = [data[0] for data in datasets]
        train_y, valid_y, test_y = [data[1] for data in datasets]

        unlabeled = load_cifar10.load_data()
        unlabeled_x = numpy.concatenate([data[0] for data in unlabeled], axis=0)
        num_unlabeled = unlabeled_x.shape[0]
        
        num_train, num_valid, num_test = [data[0].shape[0] for data in datasets]
        num_class = numpy.max(train_y) + 1

        
    xmean = train_x.mean(axis=0)
    train_x -= xmean
    valid_x -= xmean
    test_x -= xmean


    model = Model(
        image_shape=image_shape,
        filter_shape=filter_shape,
        num_class=num_class,
        doubleconv=doubleconv,
        kernel_size=kernel_size,
        kernel_pool_size=kernel_pool_size
    )

    if loadfrom:
        print 'loading model...'
        model.load_model(loadfrom)

    fn = model.build_model()


    if train_on_valid:
        num_train += num_valid
        train_x = numpy.concatenate([train_x, valid_x], axis=0)
        train_y = numpy.concatenate([train_y, valid_y], axis=0)
        valid_x = test_x
        valid_y = test_y
        num_valid = num_test

    n_train_batches = num_train / batch_size
        
    print 'training ...'

    train_errors = []
    valid_errors = []
    test_errors = []
    best_valid = 1.
    best_valid_edix = 0
    bad_count = 0
    best_model = model

    for eidx in range(train_epochs):
        c = []
        costs = []
        idx_perm = numpy.random.permutation(num_train)
        for batch_index in range(n_train_batches):
            this_idx = idx_perm[batch_index * batch_size: (batch_index + 1) * batch_size]
            this_x = train_x[this_idx]
            this_y = train_y[this_idx]
            error, cost = fn['train'](this_x, this_y, lr)
            c = numpy.append(c, error)
            costs = numpy.append(costs, cost)
            
        train_error = numpy.mean(c)
        train_errors.append(train_error)
        c = []
        for batch_index in range(num_valid / batch_size):
            this_x = valid_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            this_y = valid_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            c = numpy.append(c, fn['test'](this_x, this_y))
        valid_error = numpy.mean(c)
        valid_errors.append(valid_error)
        if train_on_valid:
            test_error = valid_error
            test_errors.append(test_error)
        else:
            c = []
            for batch_index in range(num_test / batch_size):
                this_x = test_x[batch_index * batch_size: (batch_index + 1) * batch_size]
                this_y = test_y[batch_index * batch_size: (batch_index + 1) * batch_size]
                c = numpy.append(c, fn['test'](this_x, this_y))
            test_error = numpy.mean(c)
            test_errors.append(test_error)
        
        if valid_error <= best_valid:
            best_valid = valid_error
            best_valid_edix = eidx
            bad_count = 0
            best_model = model
        else:
            bad_count += 1
            if bad_count > patience:
                print 'reducing learnig rate!'
                lr *= 0.5
                bad_count = 0
        print 'epoch ', eidx, train_error, valid_error, test_error


    if train_errors:
        print 'best errors:', train_errors[best_valid_edix], best_valid, test_errors[best_valid_edix]

    if saveto:
        print 'saving model ...'
        best_model.save_model(saveto)

    log_fp = open('logs/' + dataset + '_log.txt', 'ab')
    for k, v in opt.items():
        if type(v) is not str:
            v = str(v)
        log_fp.write(k + ': ' + v + '\n')
        
    numpy.savetxt(log_fp, numpy.array([numpy.arange(len(train_errors)), train_errors, valid_errors, test_errors]).T,
                  fmt=['%d', '%.4f', '%.4f', '%.4f'],
                  delimiter='\t')
    if train_epochs > 0:
        log_fp.write('%.4f %.4f %.4f\n' % (train_errors[best_valid_edix], best_valid, test_errors[best_valid_edix]))
    log_fp.close()

    # visualization
    W = model.params[0].get_value(borrow=True)
    W = W.reshape((W.shape[0], numpy.prod(W.shape[1:])))
    save_images(W, './plots/' + dataset + '_filters.png', filter_shape[0][-2:], color=color)

