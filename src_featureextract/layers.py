from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

# Convolution followed by pooling. Convolution always uses stride of 1, to
# generate output of the same size as the input for easier reversibility.
# The stride for pooling is the same as the kernel size, for easier reversibility.
class ConvPool(object): 
  def __init__(self, scope="conv_pool", conv_kernel_size=None, conv_output_channels=None,
               pool_size=None, nonlinearity=tf.identity, w_init=None, b_init=None):
    self.scope = scope
    self.conv_kernel_size = conv_kernel_size
    self.conv_output_channels = conv_output_channels
    self.pool_size = pool_size
    self.nonlinearity = nonlinearity
    self.w_init = w_init
    self.b_init = b_init

  def __call__(self, x):
    with tf.variable_scope(self.scope) as scope:
      self.conv_b = tf.get_variable('b_conv', initializer=tf.constant(self.b_init))
      self.conv_W = tf.get_variable('W', initializer=tf.constant(self.w_init))
      return ConvPool.forward(x, self.conv_W, self.conv_b, 
                               self.conv_kernel_size, self.pool_size, self.nonlinearity)

  @staticmethod
  def forward(x_in, conv_W, conv_b, conv_ksize, pool_size, nonlinearity):
    x_conv = tf.nn.bias_add(tf.nn.conv2d(x_in, conv_W, [1, 1, 1, 1], "SAME"), conv_b)
    x_pool = tf.nn.max_pool(nonlinearity(x_conv), [1, pool_size, pool_size, 1], 
                           [1, pool_size, pool_size, 1], "SAME")
    return x_pool

# Similar to ConvPool -- implements deconv as convolution; unpool as upsampling using
# bilinear interpolation
class DeconvUnpool(object): 
  def __init__(self, scope="conv_pool", conv_kernel_size=None, conv_output_channels=None,
               pool_size=None, nonlinearity=tf.identity):
    self.scope = scope
    self.conv_kernel_size = conv_kernel_size
    self.conv_output_channels = conv_output_channels
    self.pool_size = pool_size
    self.nonlinearity = nonlinearity

  def __call__(self, x):
    with tf.variable_scope(self.scope) as scope:
      self.conv_b = tf.get_variable('b_deconv', shape=[self.conv_output_channels],
                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
      scope.reuse_variables()
      self.conv_W = tf.get_variable('W', shape=[self.conv_kernel_size, 
                    self.conv_kernel_size, self.conv_output_channels, 
                    x.get_shape()[-1].value], 
                    initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
      print self.conv_W.name
      print self.conv_b.name
      return DeconvUnpool.forward(x, tf.transpose(self.conv_W, perm=[0, 1, 3, 2]), 
                                  self.conv_b, self.conv_kernel_size, self.pool_size,
                                  self.nonlinearity)

  @staticmethod
  def forward(x_in, conv_W, conv_b, conv_ksize, pool_size, nonlinearity):
    orig_height = x_in.get_shape()[1].value
    orig_width = x_in.get_shape()[2].value
    if pool_size > 1:
      new_height = orig_height * pool_size
      new_width = orig_width * pool_size
      x_unpool = tf.image.resize_images(x_in, [new_height, new_width])
    else:
      x_unpool = tf.identity(x_in)
    x_conv = tf.nn.bias_add(tf.nn.conv2d(nonlinearity(x_unpool), conv_W, [1, 1, 1, 1], "SAME"),
                            conv_b)
    return x_conv

class Dense_NoShare():
    """Fully-connected layer"""
    def __init__(self, scope="dense_layer", size=None, dropout=1.,
                 nonlinearity=tf.identity):
        # (str, int, (float | tf.Tensor), tf.op)
        assert size, "Must specify layer size (num nodes)"
        self.scope = scope
        self.size = size
        self.dropout = dropout # keep_prob
        self.nonlinearity = nonlinearity

    def __call__(self, x):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        # tf.Tensor -> tf.Tensor
        with tf.name_scope(self.scope):
            while True:
                try: # reuse weights if already initialized
                    return self.nonlinearity(tf.matmul(x, self.w) + self.b)
                except(AttributeError):
                    self.w, self.b = self.wbVars(x.get_shape()[1].value, self.size)
                    self.w = tf.nn.dropout(self.w, self.dropout)

    @staticmethod
    def wbVars(fan_in, fan_out):
        """Helper to initialize weights and biases, via He's adaptation
        of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
        """
        # (int, int) -> (tf.Variable, tf.Variable)
        stddev = tf.cast((2 / fan_in)**0.5, tf.float32)

        initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
        initial_b = tf.zeros([fan_out])

        return (tf.Variable(initial_w, trainable=True, name="weights"),
                tf.Variable(initial_b, trainable=True, name="biases"))


class Dense_Share(object):
  """Fully-connected layer"""
  def __init__(self, scope="dense_layer", size=None, dropout=1.,
         nonlinearity=tf.identity, w_init=None, b_init=None, isdecoder=False):
    self.scope = scope
    self.size = size
    self.isdecoder = isdecoder
    self.dropout = dropout # keep_prob
    self.nonlinearity = nonlinearity

  def __call__(self, x):
    """Dense layer currying, to apply layer to any input tensor `x`"""
    # tf.Tensor -> tf.Tensor
    with tf.variable_scope(self.scope) as scope:
      if self.isdecoder:
        self.b = tf.get_variable('b_dec', shape=[self.size])
        scope.reuse_variables()
        self.w = tf.get_variable('W', shape=[self.size, x.get_shape()[1].value],
                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        return self.nonlinearity(tf.matmul(x, tf.transpose(self.w)) + self.b)
      else:
        self.b = tf.get_variable('b_enc', shape=[self.size], initializer=
                                 tf.contrib.layers.xavier_initializer(uniform=False))
        self.w = tf.get_variable('W', shape=[x.get_shape()[1].value, self.size], initializer=
                                 tf.contrib.layers.xavier_initializer(uniform=False))
        print self.b.name
        print self.w.name
        return self.nonlinearity(tf.matmul(x, self.w) + self.b)
  
  @staticmethod
  def wbVars(fan_in, fan_out):
    """Helper to initialize weights and biases, via He's adaptation
    of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
    """
    # (int, int) -> (tf.Variable, tf.Variable)
    stddev = tf.cast((2 / fan_in)**0.5, tf.float32)

    initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
    initial_b = tf.zeros([fan_out])

    return (tf.Variable(initial_w, trainable=True, name="weights"),
        tf.Variable(initial_b, trainable=True, name="biases"))


class Layers(object):
  @staticmethod
  def conv_pool(x_in, conv_W, conv_b, conv_ksize, pool_size):
    pad_size = int((conv_ksize - 1)/2)
    x_pad = tf.pad(x_in, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    x_conv = tf.nn.conv2d(x_pad, conv_W, [1, 1, 1, 1], "VALID")
    x_pool = tf.nn.max_pool(tf.tanh(x_conv), [1, pool_size, pool_size, 1], 
                           [1, pool_size, pool_size, 1], "VALID")
    return x_pool

  @staticmethod
  def conv_unpool(x_in, conv_W, conv_b, conv_ksize, pool_size):
    orig_height = x_in.get_shape()[1].value
    orig_width = x_in.get_shape()[2].value
    new_height = orig_height * pool_size
    new_width = orig_width * pool_size

    pad_size = int((conv_ksize - 1)/2)
    x_pad = tf.pad(x_in, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    x_conv = tf.nn.conv2d(x_pad, conv_W, [1, 1, 1, 1], "VALID")
    x_unpool = tf.image.resize_images(tf.tanh(x_conv), [new_height, new_width])
    return x_unpool


