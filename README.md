Dependencies: theano lasagne (both bleeding edge), optionally cudnn.

Download the CIFAR-10 and CIFAR-100 datasets (python version) from https://www.cs.toronto.edu/~kriz/cifar.html, and extract to the data folder.

To test the dependencies, type python main.py in bash, this should start training a tiny model on cifar-10.

More explanations about the code:

The general usage of the code is python main.py [-param value]
Possible parameters are:

-dataset: the dataset to use, current supports cifar10 and cifar100, default is cifar10

-train_epochs: number of epochs to train, default is 100.

-patience: the patience for training, when the validation error stops decreasing for this number of epochs, devide learning rate by half. default is 10.

-lr: the learning rate for adadelta, default is 1.

-filter_shape: specifies the shape of the cnn. More specifically, this is the shape of the pysically allocated parameters, the actual shape is determined by the architecture.
For example:
python main.py -filter_shape 64,3,3 2,2 64,3,3 2,2 64,3,3 2,2 specifies a cnn with three conv layers, each with 64 filters and filter size 3x3, 'same' mode of convolution is used, which produces image of the same spatial size. 2,2 specifies the 2x2 max pooling. Dropout of rate 0.5 is attached immediately after each pooling layer. The last layer by default is the global average pooling layer followed by softmax. Each conv layer is also combined with batch normalization.

-conv_type: {'standard', 'double', 'maxout', 'cyclic'}.

If standard, the standard cnn is chosen, the layer shape is the same as the filter_shape.

If double, the double cnn is chosen, the layer shape is determined by the -kernel_size as well as the -kernel_pool_size together.

If maxout, maxout is used (see Maxout Networks, Goodfellow et. al.). This corresponds to performing feature pooling. The pooling size is kernel_pool_size^2 (to be consistent with double cnn).
If cyclic, cyclic cnn is chosen (see Exploiting Cyclic Symmetry in Convolutional Neural Networks, Dieleman et. al.). Maxpooling across the rotated feature maps are used, this corresponds to the "cyclic pooling" operation in the paper.

-kernel_size: the actual filter size for doulbe convolution, only odd size is supported. For example:
python main.py -filter_shape 64,4,4, 2,2 64,4,4 2,2 64,4,4 2,2 -doubleconv 1 -kernel_size 3 will allocate three conv layers with shape 64x4x4, however, the effective filter_shape that is used is 64x3x3, one is thus able to extract 4 filters of shape 64x3x3 out of each 64x4x4 by convolution. Default value is 3.

-kernel_pool_size: how to pool the activations along the large filters. As in the previous example, if kernel_pool_size is 1, the 4 64x3x3 filters are concatenated. if kernel_pool_size is 2, we take the pooling of size (2,2) among the 4 64x3x3 filters. For simplicy, -1 indicates global max pooling, which is equal to the effect of 2 in this case. default is -1. This also used by maxout cnn, where the size of a feature map is reduced by the factor of kernel_pool_size^2 times.

-batch_size: batch size of data, default is 200.

-load_model: 0 or 1. deafult 0.

-save_model: the name with which to save the model. 

-train_on_valid: whether to fold the validation set into the training set. default 1.

The training log is automatically saved at the ./logs directory, named by the dataset.

Examplar experiments:

1. python main.py -dataset cifar10 -conv_type standard -filter_shape 128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2

  This should yield an error rate of 10+%

2. python main.py -dataset cifar10 -conv_type double -filter_shape 128,4,4 128,4,4 2,2 128,4,4 128,4,4 2,2 128,4,4 128,4,4 2,2 -kernel_size 3 -kernel_pool_size -1

  This should produce a model with the same shape (w.r.t. #layers and #neurons per layer), and a corresponding error rate of ~9%.

3. python main.py -dataset cifar10 -conv_type maxout -filter_shape 512,3,3 512,3,3 2,2 512,3,3 512,3,3 2,2 512,3,3 512,3,3 2,2 -kernel_pool_size 2

   This should run a maxout network with effecitve layer size of 128 (512 / 2^2)

4. python main.py -dataset cifar10 -conv_type cyclic -filter_shape 128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2

   This runs a cyclic cnn with the same effective layer size as above.
  

