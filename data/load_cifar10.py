import cPickle as pkl
import theano
import numpy as np

data_dir = '../data/cifar-10-batches-py/'
train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4']
valid_file = 'data_batch_5'
test_file = 'test_batch'

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pkl.load(f)
    dict['data'] = np.asarray(dict['data'], dtype=theano.config.floatX)
    return dict

def load_data():
    train_data = [unpickle(data_dir + file) for file in train_files]
    valid_data = unpickle(data_dir + valid_file)
    test_data = unpickle(data_dir + test_file)

    train_set_x = np.concatenate([batch['data'] for batch in train_data], axis=0) / 255.
    train_set_y = np.concatenate([batch['labels'] for batch in train_data], axis=0).astype('int32')

    valid_set_x = valid_data['data'] / 255.
    valid_set_y = np.array(valid_data['labels']).astype('int32')

    test_set_x = test_data['data'] / 255.
    test_set_y = np.array(test_data['labels']).astype('int32')

    return ((train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y))
    