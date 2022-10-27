#coding=utf-8
import tensorflow.compat.v1 as tf
from  tensorflow.python import debug as tfdbg
tf.disable_v2_behavior()
tf.set_random_seed(seed=3543216)
import numpy as np 
import  pickle
import time
from TransX import TransEModel, TransXConfig
from RecExpert import RecExpert, TryConfig


class Generator(object):
    
    def __init__(self, tr_gen, sampled_num, batch_size,vocab_size, embedding_size, dropout_keep_prob=1.0,l2_reg_lambda=0.0,paras=None,learning_rate=0.00001,embeddings=None,loss="pair",trainable=True):
        self.model_type="Gen"
        self.l2_reg_lambda = l2_reg_lambda
        self.sampled_num = sampled_num
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.doc_nums = 737
        self.hidden_size = 512

        self.input_data1 = tf.compat.v1.placeholder(tf.int32, shape=[None, None])
        self.input_data2 = tf.compat.v1.placeholder(tf.int32, shape=[None, None])
        print(self.input_data1.shape, self.input_data2.shape)
        # self.embeddings = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([config.vocab_size, config.embed_size]),trainable = True)
        self.inputs1 = tf.nn.embedding_lookup(tr_gen.ent_embeddings, self.input_data1)
        self.inputs2 = tf.nn.embedding_lookup(tr_gen.ent_embeddings, self.input_data2)
        print(self.inputs1.shape)

        self.num_steps = tf.compat.v1.placeholder(tf.int32, shape=[None])
        print(self.num_steps)

        # self.truth = tf.placeholder(tf.float32, shape=(config.doctor_num,))
        self.truth = tf.compat.v1.placeholder(tf.float32, shape=[None, self.doc_nums])
        print(self.truth)

        # print(self.output1.shape,self.output2.shape)
        # print(self.state.shape)
        # print(self.state.h.shape)
        self.output1, self.output2 = self.getRepresentation(self.inputs1, self.inputs2, self.num_steps)

        self.keep_prob = tf.placeholder('float')
        state_drop1 = tf.nn.dropout(self.output1[:, -1, :], self.keep_prob, seed=3543216)
        state_drop2 = tf.nn.dropout(self.output2[:, -1, :], self.keep_prob, seed=3543216)

        self.state_drop = tf.concat([state_drop1, state_drop2], 1)
        print('state_drop shape is:-----------------------')
        print(self.state_drop.shape)
        # w = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False,seed = 3543216)([config.hidden_size*2, config.doctor_num]),trainable=True)
        self.w = tf.Variable(tf.glorot_normal_initializer(seed=3543216)([self.hidden_size * 2, self.doc_nums]),
                             name='Rec_weight', trainable=True)
        # b = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False,seed = 3543216)([config.doctor_num]), trainable=True)
        self.b = tf.Variable(tf.glorot_normal_initializer(seed=3543216)([self.doc_nums]), name='Rec_bias',
                             trainable=True)

        self.y = tf.matmul(self.state_drop, self.w) + self.b
        # self.y = self.y - tf.reduce_max(self.y)
        self.y_soft = tf.nn.softmax(self.y)
        print('y_soft shape')
        print(self.y.shape)

        self.cro = tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.truth)
        # print ('cross entrophy',self.cro.shape)
        self.rec_loss = tf.reduce_mean(self.cro)

        with tf.name_scope('gen_input'):
            self.gen_input_x_2 = tf.compat.v1.placeholder(tf.int32, shape=[None, self.sampled_num],name='gen_doc_input')  # [batch,10],1 positive,9 negtive
            self.reward  =tf.compat.v1.placeholder(tf.float32, shape=[None,self.sampled_num], name='reward')
            #self.neg_index  =tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='neg_index')
        #self.check_op = tf.compat.v1.add_check_numerics_ops()
        with tf.name_scope('generator_embedding'):
            print('generator random bias embedding......')
            self.gen_bias_b = tf.Variable(tf.constant(0.0, shape=[self.doc_nums]), name='gen_random_bias', trainable=True)
            #self.gen_weight_w = tf.Variable(
            #    tf.glorot_normal_initializer(seed=3543216)([self.sampled_num, self.sampled_num]),
            #    name='random_weight', trainable=True)
        self.gen_user_embedding = tf.nn.embedding_lookup(tr_gen.doc_embeddings,self.gen_input_x_2)#[batch,20,1024]
        self.gen_i_bias_b = tf.nn.embedding_lookup(self.gen_bias_b,self.gen_input_x_2)#[batch,20]
        self.l2_loss = tf.constant(0.0)
        self.l2_loss = (self.l2_loss+tf.nn.l2_loss(self.gen_user_embedding)+tf.nn.l2_loss(self.gen_i_bias_b))

        with tf.name_scope("gen_output"):
            self.all_logits = tf.matmul(self.state_drop, tr_gen.doc_embeddings, transpose_b=True) + self.gen_bias_b
            #shape[batch,737]

            self.softmax_all_logits = tf.nn.softmax(self.all_logits) #[batch,737]
            self.prob = tf.batch_gather(self.softmax_all_logits,self.gen_input_x_2) #[batch,20]
            #self.prob = tf.reshape(tf.gather_nd(self.all_logits,self.neg_index),[-1,self.sampled_num])
            print ('this is generator prob')
            print (self.prob.shape)
            #self.log_prob = tf.log(1e-10+self.prob)
            self.pre_losses = - tf.reduce_mean(tf.log(1e-10 +self.prob) *self.reward) + self.l2_loss
            self.gen_loss = self.rec_loss + self.pre_losses
            #self.gan_loss = self.pre_losses + self.loss_all

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate, name='Gen_op')
        self.grads_and_vars = optimizer.compute_gradients(self.gen_loss)
        #self.grads_and_vars = optimizer.compute_gradients(self.gan_loss)
        print('generator grads and vars------------------')
        #print((grad,var) for grad,var in self.grads_and_vars if grad is not None)
        self.show_gradvars = [(grad,var) for grad,var in self.grads_and_vars if grad is not None]
        #self.show_grad = [grad for grad,var in self.grads_and_vars if grad is not None]
        #self.show_var = [var for grad,var in self.grads_and_vars if grad is not None]
        #self.show1_gradvars = [var for grad,var in self.grads_and_vars if grad is None]
        #self.capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.grads_and_vars if grad is not None]
        self.gan_updates = optimizer.apply_gradients(self.show_gradvars, global_step=self.global_step)

    def getRepresentation(self, input1, input2, numstep):
        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(512)
            self.output1, self.state1 = tf.compat.v1.nn.dynamic_rnn(  # sequence length=60
                cell=cell,
                inputs=input1,
                dtype=tf.float32,
                sequence_length=numstep
            )
            self.output2, self.state2 = tf.compat.v1.nn.dynamic_rnn(
                cell=cell,
                inputs=input2,
                dtype=tf.float32,
                sequence_length=numstep
            )
            return self.output1, self.output2
      



