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

all_file = open('../re/q-train-test-dev_alter.pkl','rb')
tr_ = open('../../data/data111.pkl','rb')
tr_data = pickle.load(tr_)
te_ = open('../re/data_test.pkl','rb')
te_data = pickle.load(te_)
de_ = open('../../data/data33.pkl','rb')
de_data = pickle.load(de_)
tr_.close()
te_.close()
de_.close()

max_sentence_length = 60
global_file_num = 0
global_file_num1 = 0

#seed = 3543216
#random.seed(seed)
#np.random.seed(seed=seed)
#tf.set_random_seed(seed)


class TryConfig(object):
    def __init__(self):
        self.embed_size = 50
        self.hidden_size = 512
        # self.vocab_size = 40005
        self.vocab_size = 45005
        self.doctor_num = 737  # 737
        self.hits_at_k = 20
        self.batch_size = 512
        self.evaluate_batch = 256


class RecExpert(object):
    def __init__(self, config,ts):
        self.input_data1 = tf.placeholder(tf.int32, shape=[None, None])
        self.input_data2 = tf.placeholder(tf.int32, shape=[None, None])
        print (self.input_data1.shape,self.input_data2.shape)
        #self.embeddings = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([config.vocab_size, config.embed_size]),trainable = True)
        self.inputs1 = tf.nn.embedding_lookup(ts.ent_embeddings, self.input_data1)
        self.inputs2 = tf.nn.embedding_lookup(ts.ent_embeddings, self.input_data2)
        print(self.inputs1.shape)

        self.num_steps = tf.placeholder(tf.int32, shape=[None])
        print (self.num_steps)

        # self.truth = tf.placeholder(tf.float32, shape=(config.doctor_num,))
        self.truth = tf.placeholder(tf.float32, shape=[None, config.doctor_num])
        print (self.truth)

        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(config.hidden_size)
        self.output1, self.state1 = tf.compat.v1.nn.dynamic_rnn(
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

        state_drop = tf.concat([state_drop1, state_drop2], 1)
        print (state_drop.shape)
        #w = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False,seed = 3543216)([config.hidden_size*2, config.doctor_num]),trainable=True)
        w = tf.Variable(tf.glorot_normal_initializer(seed=3543216)([config.hidden_size*2, config.doctor_num]),trainable=True)
        #b = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False,seed = 3543216)([config.doctor_num]), trainable=True)
        b = tf.Variable(tf.glorot_normal_initializer(seed=3543216)([config.doctor_num]),trainable=True)

        self.y = tf.matmul(state_drop, w) + b
        self.y_soft = tf.nn.softmax(self.y)
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
	#self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.truth, logits=self.y))
        #self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)


def asarray_int32(x):
    return np.asarray(x, 'int32')


def evaluate(ol,obj1,epo, obj2,transConfig, transModel, recModel, reConfig, session):
    results_list = []
    sim_list = []
    truth_count = 0
    positive_truth_count = 0
    epoch = epo    

    tiao = []

    test = FormatData(obj1)
    test_data = test.get(obj2)
    #np.random.shuffle(test_data)
    for batch_index in range(int(len(test_data) / reConfig.batch_size)):
        bac, sent,sent_entity, sent_len, ph, pt, pr, nh, nt, nr, ground_truth = test.get_batch(reConfig.batch_size,test_data)

        #truth_list = []
        #truth_list.append(ground_truth)
        feed_dict = {
            #sgModel.train_inputs: word_x,
            #sgModel.train_labels: word_y,
	    transModel.pos_h: ph,
	    transModel.pos_t: pt,
	    transModel.pos_r: pr,
	    transModel.neg_h: nh,
	    transModel.neg_t: nt,
	    transModel.neg_r: nr,
            recModel.input_data1: sent,
	    recModel.input_data2: sent_entity,
            recModel.truth: bac,
            recModel.num_steps: sent_len,
            recModel.keep_prob: 1.0,
        }
        # print(truth_list)
        sim = session.run([recModel.y_soft], feed_dict=feed_dict)[0]
        sim_list.extend(sim)
        # print (sim)
        sim = np.argsort(-sim, axis=1)
        # print (sim)
        for i in range(0, len(sim)):
            rs1 = []
            i_posi_count = 0
            for j in range(0, 20):
                rs1.append(0)
            for k in range(0, len(rs1)):
                if sim[i][k] in ground_truth[i]:
                    i_posi_count += 1
                    positive_truth_count += 1
                    rs1[k] = 1
            # print(rs1)
            tiao.append((i, i_posi_count * 1.0 / len(ground_truth[i]), mean_average_precision([rs1]), mean_reciprocal_rank([rs1])))
            truth_count += len(ground_truth[i])
            results_list.append(rs1)

    # print(positive_truth_count, truth_count)
    # print (results_list)
    
    print('recall', positive_truth_count * 1.0 / truth_count)
    print('MAP', mean_average_precision(results_list))
    print('MRR', mean_reciprocal_rank(results_list))
    rec = positive_truth_count * 1.0 / truth_count
    ma = mean_average_precision(results_list)
    mr = mean_reciprocal_rank(results_list)
    print(' evaluate end...')

    global global_file_num
    global global_file_num1
    with open('./2021/results/2021.res_2.pkl.' + str(global_file_num), 'wb') as ff:
        pickle.dump((rec,ma,mr,ol), ff)
        global_file_num += 1
    with open('./2021/metrics/tiao.pkl.' + str(epoch) + '.' + str(global_file_num1), 'wb') as ff1:
        pickle.dump(tiao,ff1)
        global_file_num1 += 1


def main(args):
    with tf.Graph().as_default():
        global all_file
        tr, te, de = pickle.load(all_file)
        #test_data = FormatData(te)
        #dev_data = FormatData(de)
        #sgModel = SkipGram()
        transConfig = TransXConfig()
        transModel = TransEModel(config=transConfig)
        reConfig = TryConfig()
        recModel = RecExpert(config=reConfig,ts=transModel)
        

        loss_all = transModel.loss + recModel.loss
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss_all)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep=50)

        sessionConfig = tf.ConfigProto()
        sessionConfig.gpu_options.per_process_gpu_memory_fraction = 0.3
        sessionConfig.gpu_options.allow_growth = True

        when = 0
        with tf.Session(config=sessionConfig) as session:
            init.run()
            print('Initialized')
            for epoch in range(50):
                when = when + 1

                num_batch = 0
                epoch_loss = 0.
                trans_loss = 0.
                rec_loss = 0.

                training = FormatData(tr)
                training_data = training.get(tr_data)
                #print (11111111)
                #np.random.shuffle(training_data)
                for batch_index in range(int(len(training_data)/reConfig.batch_size)):
                    num_batch += 1
                    bac, sent, sent_entity, sent_len, ph, pt, pr, nh, nt, nr, ground_truth = training.get_batch(reConfig.batch_size,training_data)
                    #if num_batch ==2:
                    #    print (bac.shape,sent.shape,sent_entity.shape,sent_len.shape,word_x.shape,word_y.shape,ph.shape)
                    feed_dict = {
                            #sgModel.train_inputs: word_x,
                            #sgModel.train_labels: word_y,
			    transModel.pos_h: ph,
			    transModel.pos_t: pt,
			    transModel.pos_r: pr,
			    transModel.neg_h: nh,
			    transModel.neg_t: nt,
			    transModel.neg_r: nr,
                            recModel.input_data1: sent,
			    recModel.input_data2: sent_entity,
                            recModel.truth: bac,
                            recModel.num_steps: sent_len,
                            recModel.keep_prob: 0.6,
                    }
                    #print (22222222222)
                    _, loss_val, tr_lo, rec_lo = session.run([optimizer, loss_all, transModel.loss, recModel.loss],feed_dict=feed_dict)
                    #print (333333333333333)
                    trans_loss += tr_lo
                    rec_loss += rec_lo
                    epoch_loss += loss_val
                   
                    #print (loss_val,tr_lo,rec_lo)
                print(trans_loss, rec_loss, trans_loss * 1.0 / (num_batch),rec_loss * 1.0 / (num_batch))
                print(epoch, 'epoch end...       average loss at epoch : ',epoch_loss * 1.0 / (num_batch))
                op_loss = epoch_loss * 1.0 / num_batch
                saver.save(session, './2020/train_model/saved_sun.model', epoch)
                print(time.asctime(time.localtime(time.time())))
                if when >= 0:
                    model_file = tf.train.latest_checkpoint('./2020/train_model/')
                    saver.restore(session, model_file)
                    #test_train_data = open('../data/sun.test_newentity.pkl')
		    #saver.restore(session,'./tmp_/saved_sun.model-8')
                    evaluate(op_loss,te,epoch, te_data, transConfig, transModel, recModel, reConfig, session)

                    # test_data = open('../data/data.test.pkl')
                    saver.restore(session, model_file)
                    #test_data = open('../data/sun.dev_newentity.pkl')
                    evaluate(op_loss,de,epoch, de_data, transConfig, transModel, recModel, reConfig, session)

                    print(time.asctime(time.localtime(time.time())))
                # saver.save(session, './tmp/saved_sun.model')

if __name__ == "__main__":
    tf.app.run()
