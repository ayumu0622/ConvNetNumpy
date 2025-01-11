import numpy as np

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        #  (N, C, H, W) x dim
        #  (F, C, HH, WW) w dim
        #  (F,)
        
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        
        C, H, W  = input_dim
        F = num_filters
        # convolution
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(F, C, filter_size, filter_size)) 
        self.params['b1'] = np.zeros((F))
      
        # hidden affine
        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes)) 
        self.params['b2'] = np.zeros((hidden_dim))
        
        # self.params['W2'] = None 
        # self.params['b2'] = None

        # output affine
        self.params['W3'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes)) 
        self.params['b3'] = np.zeros((num_classes))

        self.counter = 0
        self.weight_scale = weight_scale
        self.hidden_dim = hidden_dim
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out_conv, cache_conv = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
        
        if self.counter == 0:
            out_conv_reshaped = out_conv.reshape((out_conv.shape[0], -1))
            FC_input_dim = out_conv_reshaped.shape[1]
            self.params['W2'] = np.random.normal(loc=0.0, scale=self.weight_scale, size=(FC_input_dim, self.hidden_dim)) 
            self.params['b2'] = np.zeros((self.hidden_dim))
        
        self.counter += 1
        out_fc1, cache_fc1 = affine_relu_forward(out_conv, self.params['W2'], self.params['b2'])
        scores, cache_fc2 = affine_forward(out_fc1, self.params['W3'], self.params['b3'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout  = softmax_loss(scores, y)
        loss = loss + self.reg * 0.5 * (np.sum(self.params['W1'] ** 2) + 
                                        np.sum(self.params['W2'] ** 2) + 
                                        np.sum(self.params['W3'] ** 2))
        dx2, dw2, grads['b3'] = affine_backward(dout, cache_fc2)
        grads['W3'] = dw2 + self.reg * self.params['W3']
        dx1, dw1, grads['b2'] = affine_relu_backward(dx2, cache_fc1)
        grads['W2'] = dw1 + self.reg * self.params['W2']
        dx_conv, dw_conv, grads['b1'] = conv_relu_pool_backward(dx1, cache_conv)
        grads['W1'] = dw_conv + self.reg * self.params['W1']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss, grads


pass
