"""The normalization model as implemented by TRF and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import numpy as np
import os
# enable just error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.01))


# Layer Normalization as defined in Transformer code
def LayerNormalization(x,scale,bias,epsilon=0):
  """Applies layer normalization."""
  mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
  variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
  norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
  return norm_x * scale + bias


val0 = 6
val1 = 512
scale = 1.0
bias  = 0.0
hidden_size = 4096

#initialize x_trf 
x_trf  = init_weights([val0,val1,hidden_size])

#input parameters
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("iter", 10, "Total number of iterations")
flags.DEFINE_string("mode","benchmark","Mode")

for i in range(FLAGS.iter):
 
    init = tf.compat.v1.global_variables_initializer()
    with tf.device('/GPU:0'):
        context_layer_gpu  = LayerNormalization(x_trf,scale,bias)   
        context_layer_gpu_gradient = tf.gradients(context_layer_gpu,x_trf)

    if FLAGS.mode == "validation":
        with tf.device('/CPU:0'):
            context_layer_cpu  = LayerNormalization(x_trf,scale,bias) 
            context_layer_cpu_gradient = tf.gradients(context_layer_cpu,x_trf)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        # sess.run returns a list of the fetches given to it
        gpu_pass = sess.run([context_layer_gpu,context_layer_gpu_gradient])
        forward_pass_gpu = gpu_pass[0]
        # tf.gradients returns a list of derivatives and we only need the derivative of one tensor so we take the first argument
        backward_pass_gpu = gpu_pass[1][0]

        if FLAGS.mode == "benchmark":
            continue
        
        cpu_pass = sess.run([context_layer_cpu,context_layer_cpu_gradient])
        forward_pass_cpu = cpu_pass[0]
        # tf.gradients returns a list of derivatives and we only need the derivative of one tensor so we take the first argument
        backward_pass_cpu = cpu_pass[1][0]
        # compute mean square error using numpy
        mse_fwd = ((forward_pass_gpu - forward_pass_cpu)**2).mean(axis=-1)
        print("Forward MSE Error: ", np.sum(mse_fwd))
        mse_bwd = ((backward_pass_gpu - backward_pass_cpu)**2).mean(axis=-1)
        print("Backward MSE Error: ", np.sum(mse_bwd))
        y_minus_dx_gpu = (forward_pass_gpu - backward_pass_gpu)
        print("y-dx: ", np.sum(y_minus_dx_gpu))

        

    
