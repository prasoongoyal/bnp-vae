from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import

import tensorflow as tf

class Layers(object):
  @staticmethod
  def conv_pool(x_in, conv_W, conv_b, conv_ksize, pool_size):
    pad_size = int((conv_ksize - 1)/2)
    x_pad = tf.pad(x_in, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    x_conv = tf.nn.conv2d(x_pad, conv_W, [1, 1, 1, 1], u"VALID")
    x_pool = tf.nn.max_pool(tf.tanh(x_conv), [1, pool_size, pool_size, 1], 
                           [1, pool_size, pool_size, 1], u"VALID")
    return x_pool

  @staticmethod
  def conv_unpool(x_in, conv_W, conv_b, conv_ksize, pool_size):
    orig_height = x_in.get_shape()[1].value
    orig_width = x_in.get_shape()[2].value
    new_height = orig_height * pool_size
    new_width = orig_width * pool_size

    pad_size = int((conv_ksize - 1)/2)
    x_pad = tf.pad(x_in, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    x_conv = tf.nn.conv2d(x_pad, conv_W, [1, 1, 1, 1], u"VALID")
    #x_pool = tf.nn.max_pool(tf.tanh(x_conv), [1, pool_size, pool_size, 1], 
    #                       [1, pool_size, pool_size, 1], "VALID")
    x_unpool = tf.image.resize_images(tf.tanh(x_conv), [new_height, new_width])
    return x_unpool



class Dense(object):
  u"""Fully-connected layer"""
  def __init__(self, scope=u"dense_layer", size=None, dropout=1.,
         nonlinearity=tf.identity):
    # (str, int, (float | tf.Tensor), tf.op)
    assert size, u"Must specify layer size (num nodes)"
    self.scope = scope
    self.size = size
    self.dropout = dropout # keep_prob
    self.nonlinearity = nonlinearity

  def __call__(self, x):
    u"""Dense layer currying, to apply layer to any input tensor `x`"""
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
    u"""Helper to initialize weights and biases, via He's adaptation
    of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
    """
    # (int, int) -> (tf.Variable, tf.Variable)
    stddev = tf.cast((2 / fan_in)**0.5, tf.float32)

    initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
    initial_b = tf.zeros([fan_out])

    return (tf.Variable(initial_w, trainable=True, name=u"weights"),
        tf.Variable(initial_b, trainable=True, name=u"biases"))
