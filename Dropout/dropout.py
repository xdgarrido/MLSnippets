
"""The dropout model as implemented by BERT/Trf and related functions."""
import sys
import random
import numpy as np
import os
# enable just error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf




def init_rand_variable(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


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
    rate = 1.0 - dropout_prob
    return tf.nn.dropout(input_tensor, rate, seed=seed)


# batch and seq size that fit into a single GPU collected from https://github.com/ROCmSoftwarePlatform/BERT#out-of-memory-issues
batch_size = 6
seq_length = 512

# number of heads for BERT base model collected from https://github.com/ROCmSoftwarePlatform/BERT#pre-trained-models
num_attention_heads = 24

# default dropout prob in BERT model collected from https://github.com/ROCmSoftwarePlatform/BERT/blob/bee6030e31e42a9394ac567da170a89a98d2062f/modeling.py#L42
attention_probs_dropout_prob = 0.1

# initialize atttention_scores
attention_probs = init_ones(
    [batch_size, num_attention_heads, seq_length, seq_length])

#input parameters
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("iter", 10, "Total number of iterations")
flags.DEFINE_string("mode","benchmark","Mode")

for i in range(FLAGS.iter):
    seed = random.randint(0, sys.maxsize)

    init = tf.compat.v1.global_variables_initializer()

    with tf.device('/GPU:0'):
        attention_probs_dropout_gpu = dropout(
            attention_probs, attention_probs_dropout_prob, seed=seed)

        attention_probs_dropout_gpu_gradient = tf.gradients(
            attention_probs_dropout_gpu, attention_probs)

    if FLAGS.mode == "validation":
        with tf.device('/CPU:0'):
            attention_probs_dropout_cpu = dropout(
                attention_probs, attention_probs_dropout_prob, seed=seed)

            attention_probs_dropout_cpu_gradient = tf.gradients(
                attention_probs_dropout_cpu, attention_probs)

    with tf.compat.v1.Session() as sess:
        sess.run(init)

        # sess.run returns a list of the fetches given to it
        gpu_pass = sess.run([attention_probs_dropout_gpu,
                             attention_probs_dropout_gpu_gradient])
        forward_pass_gpu = gpu_pass[0]
        # tf.gradients returns a list of derivatives and we only need the derivative of one tensor so we take the first argument
        backward_pass_gpu = gpu_pass[1][0]

        if FLAGS.mode == "benchmark":
            continue

        cpu_pass = sess.run(
            [attention_probs_dropout_cpu, attention_probs_dropout_cpu_gradient])
        forward_pass_cpu = cpu_pass[0]
        backward_pass_cpu = cpu_pass[1][0]

        print("Seed: ", seed)
        forward_error = forward_pass_gpu-forward_pass_cpu
        print("Forward Error: ", np.sum(forward_error))
        backward_error = backward_pass_gpu-backward_pass_cpu
        print("Backward Error: ", np.sum(backward_error))
        y_minus_dx = forward_pass_gpu - backward_pass_gpu
        print("y-dx: ", np.sum(y_minus_dx))
