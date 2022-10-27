# coding=utf-8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(seed = 3543216)
import numpy as np
import time
import pickle
from QACNN import QACNN
# from TransX import TransEModel, TransXConfig
from RecExpert import RecExpert, TryConfig
class Discriminator(object):

    def __init__(self, tr_dis, sampled_num, embedding_size, dropout_keep_prob=1.0, l2_reg_lambda=0.0, learning_rate=1e-2, paras=None, embeddings=None, loss="pair",trainable=True):
        self.model_type = "Dis"
        self.embedding_size = embedding_size  # hidden size,512
        self.doc_nums = 737
        self.l2_reg_lambda = l2_reg_lambda
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
            #self.input_x_1 = tf.compat.v1.placeholder(tf.int32, [None],name='dis_question_input')  # Dis model input, n-grams words,[batch]
            self.input_x_w = tf.compat.v1.placeholder(tf.int32, [None,None], name='word_input')

            #
            #self.input_x_e = tf.compat.v1.placeholder(tf.int32, [None,None], name='word_entity_input')
            #

            self.input_x_w_mask = tf.compat.v1.placeholder(tf.float32, [None, None], name='word_input_mask')

            #
            #self.input_x_e_mask = tf.compat.v1.placeholder(tf.float32, [None, None], name='word_entity_input_mask')
            #
            self.input_x_2_w = tf.compat.v1.placeholder(tf.int32, [None, 60, self.sampled_num], name='doc_input_w')#[batch, 60, 20]

            #
            #self.input_x_2_e = tf.compat.v1.placeholder(tf.int32, [None, 60, self.sampled_num], name='doc_input_e')#[batch, 60, 20]
            #
            # self.i = tf.compat.v1.placeholder(tf.int32, [None], name='bias')
            self.truth_label_w = tf.compat.v1.placeholder(tf.float32, shape=[None, 60, self.sampled_num])  # [batch, 60, 20]
            self.truth_label_w_mask = tf.boolean_mask(self.truth_label_w,tf.tile(tf.expand_dims(self.input_x_w_mask,axis=2),[1,1,sampled_num]))
            #self.truth_label_w_mask = tf.multiply(self.truth_label_w, tf.reshape(self.input_x_w_mask, [-1, 60, 1]))

            #
            #self.truth_label_e = tf.compat.v1.placeholder(tf.float32, shape=[None, 60, self.sampled_num])  # [batch, 60, 20]
            #self.truth_label_e_mask = tf.boolean_mask(self.truth_label_e,tf.tile(tf.expand_dims(self.input_x_e_mask,axis=2),[1,1,sampled_num]))
            #
            #self.truth_label_e_mask = tf.multiply(self.truth_label_e,tf.reshape(self.input_x_e_mask,[-1,60,1]))

        with tf.name_scope('discriminator_embedding'):
            print('discriminator random embedding......')
            #self.bias_b = tf.Variable(tf.random_uniform([self.sampled_num], -1.0, 1.0), name='random_bias',trainable=True)
            self.bias_b_w = tf.Variable(tf.constant(0.0, shape=[self.doc_nums]), name='random_bias', trainable=True)
            #
            #self.bias_b_e = tf.Variable(tf.constant(0.0, shape=[self.doc_nums]), name='random_bias', trainable=True)
            #

        self.w_embedding = tf.nn.embedding_lookup(tr_dis.ent_embeddings, self.input_x_w)#word embedding-[batch, 60, 50]
        #self.w_embedding = tf.multiply(self.w_embedding, tf.reshape(self.input_x_w_mask, [-1, 60, 1]))  # mask
        #
        #self.e_embedding = tf.nn.embedding_lookup(tr_dis.ent_embeddings, self.input_x_e)#word and entity embedding-[batch, 60,. 50]
        #
        #self.e_embedding = tf.multiply(self.e_embedding, tf.reshape(self.input_x_e_mask, [-1, 60, 1]))  # mask
        self.doc_embedding_w = tf.nn.embedding_lookup(tr_dis.doc_embeddings, self.input_x_2_w)#shpae-[batch,sampled_num,doc embdding_size]-[batch,60,20,50]
        #
        #self.doc_embedding_e = tf.nn.embedding_lookup(tr_dis.doc_embeddings, self.input_x_2_e) #[batch,60,20,50]
        #
        self.i_bias_b_w = tf.nn.embedding_lookup(self.bias_b_w, self.input_x_2_w)#[batch,60,20]
        #
        #self.i_bias_b_e = tf.nn.embedding_lookup(self.bias_b_e, self.input_x_2_e)#[batch,60,20]
        #
        #print('this is DIS bias_b shape-------------------')
        #print(self.i_bias_b_w.shape, self.i_bias_b_e)
        self.l2_loss = tf.constant(0.0)
        #self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.w_embedding) + tf.nn.l2_loss(self.e_embedding) + tf.nn.l2_loss(self.doc_embedding_w) + tf.nn.l2_loss(self.doc_embedding_e) + tf.nn.l2_loss(self.i_bias_b_w) + tf.nn.l2_loss(self.i_bias_b_e)
        self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.w_embedding) + tf.nn.l2_loss(
            self.doc_embedding_w) + tf.nn.l2_loss(
            self.i_bias_b_w)

        with tf.name_scope("output"):
            # self.ds = self.getRepresentation_general_d(self.input_x_2)
            # self.ds_transpose = tf.transpose(self.ds, (0,2,1))
            self.pre_w = tf.reshape(tf.tile(self.w_embedding, [1, 1, self.sampled_num]),[-1, 60, self.sampled_num, self.embedding_size])  # [batch,60,20,50]
            #
            #self.pre_e = tf.reshape(tf.tile(self.e_embedding, [1, 1, self.sampled_num]),[-1, 60, self.sampled_num, self.embedding_size])  # [batch,60,20,50]
            #
            self.scores_w = tf.reduce_sum(tf.multiply(self.doc_embedding_w, self.pre_w),axis=3) + self.i_bias_b_w #[batch,60,20]
            self.scores_w_mask = tf.boolean_mask(self.scores_w,
                                                                 tf.tile(tf.expand_dims(self.input_x_w_mask, axis=2),
                                                                         [1, 1, sampled_num]))
            #self.scores_w_mask = tf.reshape(tf.boolean_mask(self.scores_w, self.input_x_w_mask),[-1, 60, self.sampled_num])
            #self.scores_w_mask = tf.multiply(self.scores_w, tf.reshape(self.input_x_w_mask, [-1, 60, 1])) #[batch,60,20]
            #
            #self.scores_e = tf.reduce_sum(tf.multiply(self.doc_embedding_e, self.pre_e),axis=3) + self.i_bias_b_e #[batch,60,20]
            #self.scores_e_mask = tf.boolean_mask(self.scores_e,
            #                                                     tf.tile(tf.expand_dims(self.input_x_e_mask, axis=2),
            #                                                             [1, 1, sampled_num]))
            #
            #self.scores_e_mask = tf.reshape(tf.boolean_mask(self.scores_e, self.input_x_e_mask), [-1, 60, self.sampled_num])
            #self.scores_e_mask = tf.multiply(self.scores_e, tf.reshape(self.input_x_e_mask, [-1, 60, 1])) #[batch,60,20]

            #--------------------------------------------------------------------------------
            #self.final_ge_scores_w = tf.nn.softmax(self.scores_w) #----------------this one or belows
            #self.scores_w = tf.exp(self.scores_w) #[batch,60,20]
            #self.z2_w = tf.reduce_sum(self.scores_w, axis=2) #[batch,60]
            #print(self.z2_w.shape)
            #self.z2_reshape_w = tf.reshape(tf.tile(self.z2_w,[1,self.sampled_num]),[-1,60,self.sampled_num]) #[batch,60,20]
            #self.final_scores_w = tf.div(self.scores_w, self.z2_reshape_w)  #[batch,60,20]
            #print(self.final_scores_w.shape)

            #self.sum_scores_w = tf.exp(tf.log(1e-10+self.final_scores_w))  #[batch,60,20]
            #print(self.sum_scores_w.shape)
            #self.z1_w = tf.reduce_sum(self.sum_scores_w,[1,2]) #[batch]
            #self.z1_reshape_w = tf.reshape(tf.tile(tf.expand_dims(self.z1_w, 1), [1,self.sampled_num*60]),[-1,60,self.sampled_num])
            #[batch,60,20]
            #print(self.z1_reshape_w.shape)
            #self.final_ge_scores_w = tf.div(self.sum_scores_w, self.z1_reshape_w) #[batch,60,20]
            #print(self.final_ge_scores_w.shape)
            #--------------------------------------------------------------------------------
            #self.final_ge_scores_e = tf.nn.softmax(self.scores_e) #----------------this one or belows
            #self.scores_e = tf.exp(self.scores_e)  # [batch,60,20]
            #self.z2_e = tf.reduce_sum(self.scores_e, axis=2)  # [batch,60]
            #print(self.z2_e.shape)
            #self.z2_reshape_e = tf.reshape(tf.tile(self.z2_e, [1, self.sampled_num]),
            #                               [-1, 60, self.sampled_num])  # [batch,60,20]
            #self.final_scores_e = tf.div(self.scores_e, self.z2_reshape_e)  # [batch,60,20]
            #print(self.final_scores_e.shape)

            #self.sum_scores_e = tf.exp(tf.log(1e-10 + self.final_scores_e))  # [batch,60,20]
            #print(self.sum_scores_e.shape)
            #self.z1_e = tf.reduce_sum(self.sum_scores_e, [1, 2])  # [batch]
            #self.z1_reshape_e = tf.reshape(
            #    tf.tile(tf.expand_dims(self.z1_e, 1), [1, self.sampled_num * 60]),
            #    [-1, 60, self.sampled_num])
            # [batch,60,20]
            #print(self.z1_reshape_e.shape)
            #self.final_ge_scores_e = tf.div(self.sum_scores_e, self.z1_reshape_e)  # [batch,60,20]
            #print(self.final_ge_scores_e.shape)
            #--------------------------------------------------------------------------------

            #self.pre_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.pre_logits, labels=self.truth_label)
            #self.pre_losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.final_ge_scores,labels=self.truth_label))+self.l2_loss
            #self.pre_losses_w = self.cross_entropy(self.final_ge_scores_w, self.truth_label_w)
            self.pre_losses_w = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores_w_mask,labels=self.truth_label_w_mask)
            #self.pre_losses_e = self.cross_entropy(self.final_ge_scores_e, self.truth_label_e)
            #
            #self.pre_losses_e = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores_e_mask,labels=self.truth_label_e_mask)
            #
            #self.w_e_plus_loss = tf.reduce_mean(self.pre_losses_w) + tf.reduce_mean(self.pre_losses_e)
            self.w_e_plus_loss = tf.reduce_mean(self.pre_losses_w)
            self.dis_loss = self.w_e_plus_loss + self.rec_loss

            # self.loss = tf.reduce_sum(self.losses) + self.l2_reg_lambda * self.l2_loss
            self.reward_w = 2.0 * (tf.sigmoid(self.scores_w) - 0.5)  # no log, [batch,60,20]
            #self.reward_w_mask = tf.multiply(self.reward_w, tf.reshape(self.input_x_w_mask, [-1, 60, 1]))
            #
            #self.reward_e = 2.0 * (tf.sigmoid(self.scores_e) - 0.5)  # no log, [batch,60,20]
            #
            #self.reward_e_mask = tf.multiply(self.reward_e, tf.reshape(self.input_x_w_mask, [-1, 60, 1]))
            # self.positive= tf.reduce_mean(self.score12)
            # self.negative= tf.reduce_mean( self.score13)

            self.all_rating_w = tf.reshape(tf.matmul(tf.reshape(self.w_embedding, [-1, self.embedding_size]), tr_dis.doc_embeddings,transpose_b=True), [-1, 60, self.doc_nums]) + self.bias_b_w
            #[batch.60,737]
            #
            #self.all_rating_e = tf.reshape(tf.matmul(tf.reshape(self.e_embedding, [-1, self.embedding_size]), tr_dis.doc_embeddings,transpose_b=True), [-1, 60, self.doc_nums]) + self.bias_b_e
            #
            #self.all_rating_w = tf.matmul(self.w_embedding, tr_dis.doc_embeddings, transpose_b=True) + self.bias_b_w #[batch,60,737]
            #self.all_rating_e = tf.matmul(self.e_embedding, tr_dis.doc_embeddings, transpose_b=True) + self.bias_b_e #[batch,60,737]
            #self.all_rating = self.all_rating_e + self.all_rating_w
            self.all_rating = self.all_rating_w

            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer_dis = tf.train.AdamOptimizer(self.learning_rate, name='Dis_op')
            self.grads_and_vars = optimizer_dis.compute_gradients(self.dis_loss)
            # self.grads_and_vars = optimizer.compute_gradients(self.pre_loss)
            print(self.grads_and_vars)
            self.capped_gvs = [(grad, var) for grad, var in self.grads_and_vars if grad is not None]
            self.train_op = optimizer_dis.apply_gradients(self.capped_gvs, global_step=self.global_step)

    def cross_entropy(self, y, y_hat):
        res = - tf.reduce_sum(y_hat * tf.log(y))
        return res

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
