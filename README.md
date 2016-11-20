Doubly Convolutional Neural Networks

Shuangfei Zhai, Yu Cheng, Weining Lu, Zhongfei Zhang

Full paper here: https://arxiv.org/abs/1610.09716


Enviroment: 

    - Dependencies: theano lasagne (both bleeding edge), optionally cudnn.
    - To test the dependencies, type python main.py in bash, this should start training a tiny model on cifar-10.


More explanations about the code:

    - The general usage of the code is python main.py [-param value]


Possible parameters are:

   -dataset: the dataset to use, current supports cifar10 and cifar100, default is cifar10

   -train_epochs: number of epochs to train, default is 100.

   -patience: the patience for training, when the validation error stops decreasing for this number of epochs, devide learning rate by half. default is 10.

   -lr: the learning rate for adadelta, default is 1.

   -filter_shape: specifies the shape of the cnn. More specifically, this is the shape of the pysically allocated parameters, the actual shape is determined by the architecture.

   -conv_type: {'standard', 'double', 'maxout'}.

    If standard, the standard cnn is chosen, the layer shape is the same as the filter_shape.

    If double, the double cnn is chosen, the layer shape is determined by the -kernel_size as well as the -kernel_pool_size together.

    If maxout, maxout is used (see Maxout Networks, Goodfellow et. al.). This corresponds to performing feature pooling. The pooling size is kernel_pool_size^2 (to be consistent with double cnn).

   -kernel_size: the actual filter size for doulbe convolution, only odd size is supported. 

   -kernel_pool_size: how to pool the activations along the large filters. 

   -batch_size: batch size of data, default is 200.

   -load_model: 0 or 1. deafult 0.

   -save_model: the name with which to save the model. 

   -train_on_valid: whether to fold the validation set into the training set. default 1.

   -dropout_rate: dropping out when training

   -learning_decay: decaying learning rate when the validation error stops decreasing 


Experiments I on cifar10 (the hyper-parameters listed are optimal ones for each setting)

1. python main.py -dataset cifar10 -conv_type standard -filter_shape 128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2 -save_model standard.npz -train_epochs=150

   This should yield an error rate around ~9.8%


2. python main.py -dataset cifar10 -conv_type double -filter_shape 128,4,4 128,4,4 2,2 128,4,4 128,4,4 2,2 128,4,4 128,4,4 2,2 128,4,4 128,4,4 2,2 -kernel_size 3 -kernel_pool_size -1 -save_model doubleconv.npz -learning_decay 0.7 -dropout_rate 0.6

   This should produce a model with the same shape (w.r.t. #layers and #neurons per layer), and a corresponding error rate of ~8.6%.


3. python main.py -dataset cifar10 -conv_type maxout -filter_shape 512,3,3 512,3,3 2,2 512,3,3 512,3,3 2,2 512,3,3 512,3,3 2,2 -kernel_pool_size 2 -save_model maxout.npz -learning_decay 0.7 -dropout_rate 0.6

   This should run a maxout network with effecitve layer size of 128 (512 / 2^2), and a corresponding error rate of ~9.6%.


Experiments II:

This set of experiments is aimed to try the effect of doubleconv in different layers (first, second, third, fourth).
Note: when -conv_type is double and filter size is less or equal than kernel_size, the corresponding layer falls back to standard convlayer.

1. python main.py -dataset cifar10 -conv_type double -filter_shape 128,4,4 128,4,4 2,2 128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2 -kernel_size 3 -kernel_pool_size -1 -save_model doubleconv_first.npz

2. python main.py -dataset cifar10 -conv_type double -filter_shape 128,3,3 128,3,3 2,2 128,4,4 128,4,4 2,2 128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2 -kernel_size 3 -kernel_pool_size -1 -save_model doubleconv_second.npz

3. python main.py -dataset cifar10 -conv_type double -filter_shape 128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2 128,4,4 128,4,4 2,2 128,3,3 128,3,3 2,2 -kernel_size 3 -kernel_pool_size -1 -save_model doubleconv_third.npz

4. python main.py -dataset cifar10 -conv_type double -filter_shape 128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2 128,3,3 128,3,3 128,4,4 128,4,4 2,2 2,2 -kernel_size 3 -kernel_pool_size -1 -save_model doubleconv_fourth.npz


Experiments III:

1. python main.py -dataset cifar10 -conv_type double -filter_shape 16,6,6 16,6,6 2,2 16,6,6 16,6,6 2,2 16,6,6 16,6,6 2,2 -kernel_size 3 -kernel_pool_size 1 -save_model doubleconv3_1.npz

  same #parameters, larger model

2. python main.py -dataset cifar10 -conv_type double -filter_shape 4,10,10 4,10,10 2,2 4,10,10 4,10,10 2,2 4,10,10 4,10,10 2,2 -kernel_size 3 -kernel_pool_size 1 -save_model doubleconv3_2.npz

  less #parameters, larger model