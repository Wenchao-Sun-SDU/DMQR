from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import numpy as np
np.random.seed(seed=3543216)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(seed=3543216)
import random
random.seed(3543216)
import pickle as pickle
import json
#from SkipGram import SkipGram
from TransX import TransEModel, TransXConfig
from FormatData_bac import FormatData
from rank_metrics import mean_reciprocal_rank, mean_average_precision

class TryConfig(object):
    def __init__(self):
        self.embed_size = 50
        self.hidden_size = 512
        # self.vocab_size = 40005
        self.vocab_size = 45005
        self.doctor_num = 737  # 737
        self.hits_at_k = 20
        self.batch_size = 256
        self.evaluate_batch = 256


class RecExpert(TransEModel):
    def __init__(self, config, trConf):
        TransEModel.__init__(self,trConf)

        self.input_data1 = tf.placeholder(tf.int32, shape=[None, None])
        self.input_data2 = tf.placeholder(tf.int32, shape=[None, None])
        print (self.input_data1.shape,self.input_data2.shape)
        #self.embeddings = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([config.vocab_size, config.embed_size]),trainable = True)
        self.inputs1 = tf.nn.embedding_lookup(self.ent_embeddings, self.input_data1)
        self.inputs2 = tf.nn.embedding_lookup(self.ent_embeddings, self.input_data2)
        print(self.inputs1.shape)

        self.num_steps = tf.placeholder(tf.int32, shape=[None])
        print (self.num_steps)

        # self.truth = tf.placeholder(tf.float32, shape=(config.doctor_num,))
        self.truth = tf.placeholder(tf.float32, shape=[None, config.doctor_num])
        print (self.truth)

        with tf.variable_scope('lstm',reuse=tf.AUTO_REUSE):
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(config.hidden_size)
            self.output1, self.state1 = tf.compat.v1.nn.dynamic_rnn(#sequence length=60
                cell=cell,
                inputs=self.inputs1,
                dtype=tf.float32,
                sequence_length=self.num_steps
            )
            self.output2, self.state2 = tf.compat.v1.nn.dynamic_rnn(
                cell=cell,
                inputs=self.inputs2,
                dtype=tf.float32,
                sequence_length=self.num_steps
            )

        print(self.output1.shape,self.output2.shape)
        # print(self.state.shape)
        # print(self.state.h.shape)

        self.keep_prob = tf.placeholder('float')
        state_drop1 = tf.nn.dropout(self.output1[:, -1, :], self.keep_prob,seed=3543216)
        state_drop2 = tf.nn.dropout(self.output2[:, -1, :], self.keep_prob,seed=3543216)

        self.state_drop = tf.concat([state_drop1, state_drop2], 1)
        print (self.state_drop.shape)
        #w = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False,seed = 3543216)([config.hidden_size*2, config.doctor_num]),trainable=True)
        w = tf.Variable(tf.glorot_normal_initializer(seed=3543216)([config.hidden_size*2, config.doctor_num]),trainable=True)
        #b = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False,seed = 3543216)([config.doctor_num]), trainable=True)
        b = tf.Variable(tf.glorot_normal_initializer(seed=3543216)([config.doctor_num]),trainable=True)

        self.y = tf.matmul(self.state_drop, w) + b
        self.y_soft = tf.nn.softmax(self.y)
        print ('y_soft shape')
        print(self.y.shape)

        #log_y_soft = tf.log(self.y_soft)
        #self.loss = -tf.reduce_sum(log_y_soft*self.truth) / config.batch_size

        #self.y_soft = tf.nn.sigmoid(self.y)
        #print(self.y.shape)
        # print(self.y_soft.shape)
        # log_y_soft = tf.log(self.y_soft)
        # self.loss = -tf.reduce_sum(tf.matmul(self.truth, log_y_soft, transpose_b = True)) / config.batch_size
        # self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.truth, logits=self.y))
        self.cro = tf.nn.softmax_cross_entropy_with_logits(logits=self.y,labels=self.truth)
        #print ('cross entrophy',self.cro.shape)
        self.loss = tf.reduce_mean(self.cro)

        self.loss_all = self.loss + self.loss_trans
        with tf.variable_scope('rec_optimizer', reuse=tf.AUTO_REUSE):
            self.rec_optimizer = tf.train.AdamOptimizer(0.001, name='rec_op').minimize(self.loss_all)
        #self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.truth, logits=self.y))
        #self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)