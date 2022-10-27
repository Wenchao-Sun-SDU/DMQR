#coding=utf-8
import tensorflow.compat.v1 as tf
from  tensorflow.python import debug as tfdbg
tf.disable_v2_behavior()
tf.set_random_seed(seed=3543216)
import numpy as np 
import time
import pickle
#from TransX import TransEModel, TransXConfig
from RecExpert import RecExpert, TryConfig
class Discriminator(object):

    def __init__(self, tr_dis, sampled_num, batch_size, vocab_size, embedding_size, dropout_keep_prob=1.0,l2_reg_lambda=0.0,learning_rate=0.00001,paras=None,embeddings=None,loss="pair",trainable=True):
        self.model_type="Dis"
        self.embedding_size = embedding_size #hidden size,512
        self.doc_nums = 737
        self.l2_reg_lambda= l2_reg_lambda
        self.sampled_num = sampled_num
        self.learning_rate = learning_rate
        self.hidden_size = 512

        self.input_data1 = tf.compat.v1.placeholder(tf.int32, shape=[None, None])
        self.input_data2 = tf.compat.v1.placeholder(tf.int32, shape=[None, None])
        print(self.input_data1.shape, self.input_data2.shape)
        # self.embeddings = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([config.vocab_size, config.embed_size]),trainable = True)
        self.inputs1 = tf.nn.embedding_lookup(tr_dis.ent_embeddings, self.input_data1)
        self.inputs2 = tf.nn.embedding_lookup(tr_dis.ent_embeddings, self.input_data2)
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

        with tf.name_scope('dis_input'):
            #self.input_x_1 = tf.compat.v1.placeholder(tf.int32, [None, None], name = 'question_input')#RecModel.state_drop:[batch,1024]
            self.input_x_2 = tf.compat.v1.placeholder(tf.int32, [None, self.sampled_num], name='doc_input')#[batch,10],1 positive,9 negtive
            #self.i = tf.compat.v1.placeholder(tf.int32, [None], name='bias')
            self.truth_label = tf.compat.v1.placeholder(tf.float32, shape=[None, self.sampled_num], name='truth-label') #[batch, 10], 1 positive label, 9 negtive label

        with tf.name_scope('discriminator_embedding'):
            print ('discriminator random embedding......')
            self.bias_b = tf.Variable(tf.constant(0.0,shape=[self.doc_nums]),name='random_bias',trainable=True)
            #self.weight_w = tf.Variable(tf.random_uniform([self.sampled_num,self.sampled_num],-1.0,1.0),name='random_weight',trainable=True)
            #self.weight_w = tf.Variable(
            #    tf.glorot_normal_initializer(seed=3543216)([self.sampled_num, self.sampled_num]),
            #    name='random_weight', trainable=True)
            #self.weight_w = tf.Variable(tf.random_uniform([self.sampled_num,self.sampled_num],-1.0,1.0),name='random_weight',trainable=True)

        self.user_embedding = tf.nn.embedding_lookup(tr_dis.doc_embeddings,self.input_x_2)#[batch,20,1024]
        self.i_bias_b = tf.nn.embedding_lookup(self.bias_b,self.input_x_2)#[batch,20]
        print('this is DIS bias_b shape-------------------')
        print(self.i_bias_b.shape)
        #self.question_embedding = tf.nn.embedding_lookup(self.Embedding_question,self.input_x_1)#[batch,input1+input2,embedding_size]
        #self.i_bias = tf.gather(self.bias_b,self.i)
        self.l2_loss = tf.constant(0.0)
        self.l2_loss = (self.l2_loss+tf.nn.l2_loss(self.user_embedding) + tf.nn.l2_loss(self.i_bias_b))
        with tf.name_scope("dis_output"):
            #self.para_1 = tf.reshape(tf.tile(self.state_drop,[self.sampled_num]),[self.sampled_num,-1])
            #print (self.para_1.shape) #[batch,20,1024]
            #self.pre_logits = tf.reduce_sum(tf.multiply(self.para_1,self.user_embedding)) + self.bias_b
            self.pre = tf.reshape(tf.tile(self.state_drop,[1,self.sampled_num]),[-1,self.sampled_num,self.embedding_size*2]) #[batch,20,1024]
            self.pre_logit = tf.reduce_sum(tf.multiply(self.user_embedding,self.pre),2) + self.i_bias_b #[batch,20]
            print('logit')
            print (self.pre_logit.shape)
            #user_embedding[batch,20,1024],state_drop[batch,1024] tf.tile(state_drop,[1,20])--[batch,1024*20],tf.reshape(vector,[-1,20,1024])
            #self.pre_logits = tf.matmul(self.pre_logit, self.weight_w)+self.bias_b
            #self.pre_logits = tf.matmul(self.pre_logit,self.weight_w) + self.i_bias_b
            #self.pre_logits = self.pre_logit + self.i_bias_b
            #self.pre_logits = tf.reduce_sum(tf.matmul(self.question_embedding, self.Embedding_doc, transpose_b = True),1) + self.bias_b #[batch,737]
            #print('logits')
            #print (self.pre_logits.shape)
            self.pre_losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pre_logit,labels=self.truth_label))+self.l2_loss
            #self.pre_loss = self.loss_all
            #self.pre_loss = self.pre_losses + self.loss_all
            self.dis_loss = self.rec_loss + self.pre_losses

            #self.losses = tf.maximum(0.0, tf.subtract(0.05, tf.subtract(self.score12, self.score13)))
            #self.loss = tf.reduce_sum(self.losses) + self.l2_reg_lambda * self.l2_loss
            #self.reward_pre = tf.reshape(tf.tile(self.state_drop,[1,self.sampled_num]),[-1,self.sampled_num,self.embedding_size*2])
            #self.reward_logit = tf.reduce_sum(tf.multiply(self.user_embedding,self.reward_pre),2)
            #self.reward_logits = tf.matmul(self.reward_logit,self.weight_w) + self.i_bias_b
            #self.reward_logits = self.reward_logit +self.i_bias_b
            #print('reward logit shape-----------------')
            #print(self.reward_logits.shape)
            self.reward = 2.0*(tf.sigmoid(self.pre_logit) -0.5) # no log
            #self.reward = tf.sigmoid(self.reward_logits)
            #self.positive= tf.reduce_mean(self.score12)
            #self.negative= tf.reduce_mean( self.score13)

            #self.all_rating = self.pre_logits
            self.all_rating = tf.matmul(self.state_drop, tr_dis.doc_embeddings, transpose_b=True) + self.bias_b #[batch,737]
            self.dis_softmax = tf.nn.softmax(self.all_rating)

            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(self.learning_rate, name='Dis_op')
            self.grads_and_vars = optimizer.compute_gradients(self.dis_loss)
            # self.grads_and_vars = optimizer.compute_gradients(self.pre_loss)
            print(self.grads_and_vars)

            self.capped_gvs = [(grad, var) for grad, var in self.grads_and_vars if grad is not None]
            self.train_op = optimizer.apply_gradients(self.capped_gvs, global_step=self.global_step)

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

