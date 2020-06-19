from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os
# enable just error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from modeling import BertConfig, BertModel

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("iter", 5000, "Total number of iterations")
flags.DEFINE_integer("batch", 6, "Batch size")
flags.DEFINE_integer("seq_length", 512, "Sequence Length")
flags.DEFINE_integer("heads",16,"HEADS")
flags.DEFINE_integer("layers",24,"LAYERS")
flags.DEFINE_string("mode","benchmark","Mode")
flags.DEFINE_string("precision","fp32","PRECISION")

# batch and seq size that fit into a single GPU collected from https://github.com/ROCmSoftwarePlatform/BERT#out-of-memory-issues
batch_size = FLAGS.batch
seq_length = FLAGS.seq_length
heads = FLAGS.heads
layers = FLAGS.layers

if FLAGS.precision == "fp32":
# this is set to LARGE Bert model 
   bert_config = BertConfig(attention_probs_dropout_prob= 0.9, # 1 - 0.9
      hidden_act= "gelu",
      hidden_dropout_prob= 0.9, # 1 - 0.9
      hidden_size = 1024,
      initializer_range = 0.02,
      intermediate_size = 4096,
      max_position_embeddings = 512,
      num_attention_heads = heads,
      num_hidden_layers = layers,
      type_vocab_size =  2,
      vocab_size = 30522,
      precision=tf.float32)
else:
   bert_config = BertConfig(attention_probs_dropout_prob= 0.9, # 1 - 0.9
      hidden_act= "gelu",
      hidden_dropout_prob= 0.9, # 1 - 0.9
      hidden_size = 1024,
      initializer_range = 0.02,
      intermediate_size = 4096,
      max_position_embeddings = 512,
      num_attention_heads = heads,
      num_hidden_layers = layers,
      type_vocab_size =  2,
      vocab_size = 30522,
      precision=tf.float16)
  

# Set the bert model input
input_ids   = tf.ones((batch_size, seq_length), dtype=tf.int32)
input_mask  = tf.ones((batch_size, seq_length), dtype=tf.int32)
token_ids = tf.ones((batch_size, seq_length), dtype=tf.int32)

# Define to define loss
labels = tf.ones((batch_size, ), dtype=tf.int32)

bert_model = BertModel(
      config=bert_config,
      is_training=True,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=token_ids,
      use_one_hot_embeddings=False)

# Finalize bert
output_layer = bert_model.get_pooled_output()
hidden_size = 1024
logits = tf.compat.v1.layers.dense(output_layer, units=hidden_size, activation=tf.nn.softmax)
# This is just to compute backward pass
loss   = tf.compat.v1.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
variables = tf.compat.v1.trainable_variables()
# Any optimizer will do it
opt        = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=2e-5)
grads      = opt.compute_gradients(loss)
bert_train = opt.apply_gradients(grads)
config = tf.compat.v1.ConfigProto()

# fire-up bert
with tf.compat.v1.Session(config=config) as sess:
  sess.run(tf.compat.v1.global_variables_initializer())
  sess.run(bert_train)
  for i in range(FLAGS.iter):
    with tf.device('/GPU:0'):
      sess.run(bert_train)