# coding=utf-8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pickle
import time
from TransX import TransEModel, TransXConfig
from QACNN import QACNN
from RecExpert import RecExpert, TryConfig


class Generator(object):

    def __init__(self, tr_gen, sampled_num, embedding_size, dropout_keep_prob=1.0, l2_reg_lambda=0.0, learning_rate=1e-2, paras=None, embeddings=None, loss="pair",trainable=True):
        # QACNN.__init__(self, trConf, sequence_length, batch_size,vocab_size, embedding_size,filter_sizes, num_filters, dropout_keep_prob=dropout_keep_prob,l2_reg_lambda=l2_reg_lambda,paras=paras,learning_rate=learning_rate,embeddings=embeddings,loss=loss,trainable=trainable)
        self.model_type = "Gen"
        self.embedding_size = embedding_size
        self.doc_nums = 737
        self.l2_reg_lambda = l2_reg_lambda
        self.sampled_num = sampled_num
        self.learning_rate = learning_rate
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
        self.output1, self.output2 = self.gen_getRepresentation(self.inputs1, self.inputs2, self.num_steps)

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
        self.gen_rec_loss = tf.reduce_mean(self.cro)

        with tf.name_scope('gen_input'):
            self.input_x_w = tf.compat.v1.placeholder(tf.int32, [None,None], name='word_input')
            #
            #self.input_x_e = tf.compat.v1.placeholder(tf.int32, [None,None], name='word_entity_input')
            #

            self.input_x_w_mask = tf.compat.v1.placeholder(tf.int32, [None, None], name='word_input_mask')
            #
            #self.input_x_e_mask = tf.compat.v1.placeholder(tf.int32, [None, None], name='word_entity_input_mask')
            #

            self.gen_input_x_2_w = tf.compat.v1.placeholder(tf.int32, [None,60, self.sampled_num], name='doc_input_w')  # [batch,60, 20]
            #
            #self.gen_input_x_2_e = tf.compat.v1.placeholder(tf.int32, [None,60, self.sampled_num], name='doc_input_e')  # [batch,60, 20]
            #

            self.reward_w = tf.compat.v1.placeholder(tf.float32, shape=[None,60, self.sampled_num], name='reward_w')
            self.reward_w_mask = tf.boolean_mask(self.reward_w,
                                                                 tf.tile(tf.expand_dims(self.input_x_w_mask, axis=2),
                                                                         [1, 1, sampled_num]))
            #self.reward_w_mask = tf.reshape(tf.boolean_mask(self.reward_w, self.input_x_w_mask),[-1,60,self.sampled_num])
            #
            #self.reward_e = tf.compat.v1.placeholder(tf.float32, shape=[None,60, self.sampled_num], name='reward_e')
            #self.reward_e_mask = tf.boolean_mask(self.reward_e,
            #                                                tf.tile(tf.expand_dims(self.input_x_e_mask, axis=2),
            #                                                        [1, 1, sampled_num]))
            #

            #self.reward_e_mask = tf.reshape(tf.boolean_mask(self.reward_e, self.input_x_e_mask),[-1,60,self.sampled_num])
            #self.neg_index = tf.compat.v1.placeholder(tf.int32, shape=[None], name='neg_index')

        with tf.name_scope('generator_embedding'):
            print('generator random embedding......')
            #self.bias_b = tf.Variable(tf.random_uniform([self.sampled_num], -1.0, 1.0), name='random_bias',trainable=True)
            self.gen_bias_b_w = tf.Variable(tf.constant(0.0, shape=[self.doc_nums]), name='random_bias', trainable=True)
            #
            #self.gen_bias_b_e = tf.Variable(tf.constant(0.0, shape=[self.doc_nums]), name='random_bias', trainable=True)
            #

        self.gen_w_embedding = tf.nn.embedding_lookup(tr_gen.ent_embeddings, self.input_x_w)  # word embedding-[batch,60, 50]
        #self.gen_w_embedding = tf.multiply(self.gen_w_embedding,tf.reshape(self.input_x_w_mask,[-1,60,1])) #mask
        #
        #self.gen_e_embedding = tf.nn.embedding_lookup(tr_gen.ent_embeddings, self.input_x_e)  # word and entity embedding-[batch,60, 50]
        #
        #self.gen_e_embedding = tf.multiply(self.gen_e_embedding,tf.reshape(self.input_x_e_mask,[-1,60,1])) #mask
        self.gen_doc_embedding_w = tf.nn.embedding_lookup(tr_gen.doc_embeddings, self.gen_input_x_2_w)  # shpae-[batch,sampled_num,doc embdding_size]-[batch,60,20,50]
        #
        #self.gen_doc_embedding_e = tf.nn.embedding_lookup(tr_gen.doc_embeddings, self.gen_input_x_2_e)
        #
        #[batch,60,20,50]
        self.gen_i_bias_b_w = tf.nn.embedding_lookup(self.gen_bias_b_w, self.gen_input_x_2_w)  # [batch,60,20]
        #
        #self.gen_i_bias_b_e = tf.nn.embedding_lookup(self.gen_bias_b_e, self.gen_input_x_2_e)  # [batch,60,20]
        #
        #print('this is GEN bias_b shape-------------------')
        #print(self.gen_i_bias_b_w.shape)
        self.l2_loss = tf.constant(0.0)
        self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.gen_w_embedding) + tf.nn.l2_loss(self.gen_doc_embedding_w) + tf.nn.l2_loss(self.gen_i_bias_b_w)
        #self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.gen_w_embedding) + tf.nn.l2_loss(self.gen_e_embedding) +tf.nn.l2_loss(self.gen_doc_embedding_w) +tf.nn.l2_loss(self.gen_doc_embedding_e) + tf.nn.l2_loss(self.gen_i_bias_b_w) + tf.nn.l2_loss(self.gen_i_bias_b_e)

        with tf.name_scope("gen_output"):
            self.gen_all_logits_w = tf.reshape(tf.matmul(tf.reshape(self.gen_w_embedding,[-1,self.embedding_size]), tr_gen.doc_embeddings, transpose_b = True),[-1,60,self.doc_nums]) + self.gen_bias_b_w #[batch,60,737]
            #
            #self.gen_all_logits_e = tf.reshape(tf.matmul(tf.reshape(self.gen_e_embedding,[-1,self.embedding_size]), tr_gen.doc_embeddings, transpose_b = True),[-1,60,self.doc_nums]) + self.gen_bias_b_e #[batch,60,737]
            #
            #self.gen_all_logits = self.gen_all_logits_w + self.gen_all_logits_e #[batch,60,737]

            #------------------------------------------------------------------------------------------
            self.gen_final_ge_scores_w = tf.nn.softmax(self.gen_all_logits_w) #[batch,60,737]
            #self.gen_all_scores_w = tf.exp(self.gen_all_logits_w) #[batch,737]
            #self.gen_z2_w = tf.reduce_sum(self.gen_all_scores_w, axis=1)  # [batch]
            #self.gen_z2_reshape_w = tf.tile(tf.expand_dims(self.gen_z2_w, 1), [1, self.doc_nums]) #[batch,737]
            #self.gen_final_scores_w = tf.div(self.gen_all_scores_w, self.gen_z2_reshape_w)  # [batch,737]
            #print(self.gen_final_scores_w.shape)

            #self.gen_sum_scores_w = tf.exp(tf.log(self.gen_final_scores_w))  # [batch,737]
            #print(self.gen_sum_scores_w.shape)
            #self.gen_z1_w = tf.reduce_sum(self.gen_sum_scores_w)
            #self.gen_final_ge_scores_w = tf.div(self.gen_sum_scores_w, self.gen_z1_w)  # [batch,737]
            #print(self.gen_final_ge_scores_w.shape)
            #-------------------------------------------------------------------------------------------
            #
            #self.gen_final_ge_scores_e = tf.nn.softmax(self.gen_all_logits_e) #[batch,60,737]
            #
            #self.gen_all_scores_e = tf.exp(self.gen_all_logits_e)  # [batch,737]
            #self.gen_z2_e = tf.reduce_sum(self.gen_all_scores_e, axis=1)  # [batch]
            #self.gen_z2_reshape_e = tf.tile(tf.expand_dims(self.gen_z2_e, 1), [1, self.doc_nums])  # [batch,737]
            #self.gen_final_scores_e = tf.div(self.gen_all_scores_e, self.gen_z2_reshape_e)  # [batch,737]
            #print(self.gen_final_scores_e.shape)

            #self.gen_sum_scores_e = tf.exp(tf.log(self.gen_final_scores_e))  # [batch,737]
            #print(self.gen_sum_scores_e.shape)
            #self.gen_z1_e = tf.reduce_sum(self.gen_sum_scores_e)
            #self.gen_final_ge_scores_e = tf.div(self.gen_sum_scores_e, self.gen_z1_e)  # [batch,737]
            #print(self.gen_final_ge_scores_e.shape)
            #-------------------------------------------------------------------------------------------
            self.prob_w = tf.batch_gather(self.gen_final_ge_scores_w, self.gen_input_x_2_w)
            #[batch,60,20]
            self.prob_w_mask = tf.boolean_mask(self.prob_w,
                                                            tf.tile(tf.expand_dims(self.input_x_w_mask, axis=2),
                                                                    [1, 1, sampled_num]))
            #self.prob_w_mask = tf.reshape(tf.boolean_mask(self.prob_w,self.input_x_w_mask),[-1,60,self.sampled_num])
            #self.prob_w_mask = tf.multiply(self.prob_w, tf.reshape(self.input_x_w_mask, [-1, 60, 1]))
            #
            #self.prob_e = tf.batch_gather(self.gen_final_ge_scores_e, self.gen_input_x_2_e)
            #self.prob_e_mask = tf.boolean_mask(self.prob_e,
            #                                                tf.tile(tf.expand_dims(self.input_x_e_mask, axis=2),
            #                                                        [1, 1, sampled_num]))
            #
            #self.prob_e_mask = tf.reshape(tf.boolean_mask(self.prob_e,self.input_x_e_mask),[-1,60,self.sampled_num])
            #self.prob_e_mask = tf.multiply(self.prob_e, tf.reshape(self.input_x_e_mask, [-1, 60, 1]))
            #print('this is generator prob')
            #print(self.prob_w_mask.shape,self.prob_e_mask.shape)
            #self.w_e_plus_loss = -tf.reduce_mean(tf.log(1e-10 + self.prob_w_mask) * self.reward_w_mask) + (-tf.reduce_mean(tf.log(1e-10 + self.prob_e_mask) * self.reward_e_mask))
            self.w_e_plus_loss = -tf.reduce_mean(tf.log(1e-10 + self.prob_w_mask) * self.reward_w_mask)
            self.gen_loss = self.w_e_plus_loss + self.gen_rec_loss

            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer_gen = tf.train.AdamOptimizer(self.learning_rate, name='Gen_op')
            self.grads_and_vars = optimizer_gen.compute_gradients(self.gen_loss)
            # self.grads_and_vars = optimizer.compute_gradients(self.gan_loss)
            print('generator grads and vars------------------')
            # print((grad,var) for grad,var in self.grads_and_vars if grad is not None)
            self.show_gradvars = [(grad, var) for grad, var in self.grads_and_vars if grad is not None]
            self.gan_updates = optimizer_gen.apply_gradients(self.show_gradvars, global_step=self.global_step)

    def gen_cross_entropy(self, y, y_hat):
        assert y.shape == y_hat.shape
        res = - np.sum(y_hat * tf.log(y))
        return res

    def gen_getRepresentation(self, input1, input2, numstep):
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

        # minize attention
        # self.gan_score=self.score13-self.score12
        # self.dns_score=self.score13




