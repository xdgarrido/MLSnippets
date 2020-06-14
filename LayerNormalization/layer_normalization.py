"""The normalization model as implemented by TRF and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import numpy as np
import os
# enable just error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#input parameters
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("iter", 5000, "Total number of iterations")
flags.DEFINE_integer("hidden_size", 1024, "Hidden Size")
flags.DEFINE_string("mode","benchmark","Mode")

def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.01))


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  # return tf.contrib.layers.layer_norm(
  #     inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
  return tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,  center=False, scale=False)(inputs=input_tensor)


hidden_size = FLAGS.hidden_size

#initialize x_trf 
x_trf  = init_weights([hidden_size,hidden_size])


for i in range(FLAGS.iter):
 
    init = tf.compat.v1.global_variables_initializer()
    with tf.device('/GPU:0'):
        context_layer_gpu  = layer_norm(x_trf)   
        context_layer_gpu_gradient = tf.gradients(ys=context_layer_gpu,xs=x_trf)

    if FLAGS.mode == "validation":
        with tf.device('/CPU:0'):
            context_layer_cpu  = layer_norm(x_trf) 
            context_layer_cpu_gradient = tf.gradients(ys=context_layer_cpu,xs=x_trf)

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

        

    
