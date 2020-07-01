
"""The dropout model as implemented by BERT/Trf and related functions."""
import sys
import random
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
flags.DEFINE_integer("batch", 6, "Batch size")
flags.DEFINE_integer("seq_length", 512, "Sequence Length")
flags.DEFINE_integer("attention_heads",16,"Number of attention heads")
flags.DEFINE_string("mode","benchmark","Mode")


def init_rand_variable(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.01))


def init_ones(shape):
    return tf.ones(shape)


def dropout(input_tensor, dropout_prob, seed):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
       dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).
      seed: Python int. Seed of Random Number Generator

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    return tf.nn.dropout(input_tensor, rate=dropout_prob, seed=seed)

class Drop(tf.test.TestCase):
 
  def test_drop(self):
    with self.session() as sess:
      with tf.device('/GPU:0'): 

        # batch and seq size that fit into a single GPU collected from https://github.com/ROCmSoftwarePlatform/BERT#out-of-memory-issues
        batch_size = FLAGS.batch
        seq_length = FLAGS.seq_length

        # number of heads for BERT base model collected from https://github.com/ROCmSoftwarePlatform/BERT#pre-trained-models
        num_attention_heads = FLAGS.attention_heads

        # default dropout prob in BERT model collected from https://github.com/ROCmSoftwarePlatform/BERT/blob/bee6030e31e42a9394ac567da170a89a98d2062f/modeling.py#L42
        attention_probs_dropout_prob = 0.1

        # initialize atttention_scores
        attention_probs = init_ones([batch_size, num_attention_heads, seq_length, seq_length])

        seed = random.randint(0, sys.maxsize)
        
        attention_probs_dropout_gpu = dropout(
            attention_probs, attention_probs_dropout_prob, seed=seed)

        attention_probs_dropout_gpu_gradient = tf.gradients(
            ys=attention_probs_dropout_gpu, xs=attention_probs)
   
        init_op = tf.group(tf.compat.v1.global_variables_initializer())
        sess.run(init_op)
        for _ in range(FLAGS.iter):
            sess.run(attention_probs_dropout_gpu_gradient)

if __name__ == "__main__":
  tf.test.main()


