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

class Drop(tf.test.TestCase):
 
  def test_drop(self):
    with self.session() as sess:
      with tf.device('/GPU:0'): 

        hidden_size = FLAGS.hidden_size

        #initialize x_trf 
        x_trf  = init_weights([hidden_size,hidden_size])
        
        context_layer_gpu  = layer_norm(x_trf)   
        context_layer_gpu_gradient = tf.gradients(ys=context_layer_gpu,xs=x_trf)
   
        init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                          tf.compat.v1.local_variables_initializer())
        sess.run(init_op)
        for _ in range(FLAGS.iter):
            sess.run(context_layer_gpu_gradient)

if __name__ == "__main__":
  tf.test.main()
        

    
