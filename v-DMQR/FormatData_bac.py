#coding=utf-8
import numpy as np
np.random.seed(seed=3543216)
import json
import random
random.seed(3543216)
import pickle as pickle
import collections

max_sentence_length = 60
max_len = 7
#seed = 3543216
#random.seed(seed)
#np.random.seed(seed=seed)

ff = open('./board.pkl','rb')
board = pickle.load(ff)
board = set(board)

def asarray_int32(x):
    return np.asarray(x, 'int32')

class FormatData(object):
    def __init__(self, data):
        self.count = 0
        self.start = 0
        #self.batch_start = 0
        self.data = data
        self.how_many = 0

        ff = open('./doctor2index.pkl','rb')
        self.doc_doc = pickle.load(ff)

        f = open('./entity2dic_40002.json')
        self.en_dic = json.loads(f.read())

        f2 = open('./word-cooccurance.txt')
        self.cooccurance = [json.loads(x) for x in f2.readlines()]

        f3 = open('./triple2id_quchong_xushang_40002.txt')
        self.all_tri = f3.readlines()
        self.ent2tripleList = collections.defaultdict(list)
        for tt in self.all_tri:
            tt_list = tt.split()
            self.ent2tripleList[int(tt_list[0])].append((int(tt_list[0]),int(tt_list[1]),int(tt_list[2])))
            self.ent2tripleList[int(tt_list[1])].append((int(tt_list[0]),int(tt_list[1]),int(tt_list[2])))

        f.close()
        f2.close()
        f3.close()

    def get(self,data1):
        al_ready = data1
        #q_str, question, d_str, ent_list = self.data[self.count]
        total_list = []
        count  = 0
        for i_data in self.data:
            q_str, question, d_str, word2num_,positive_list,ent_list = i_data

            count = count +1
            word_x = []
            word_y = []

            negative_list = []

            for i in positive_list:
                neg = []
                while len(neg)<3:
                    rep = random.randint(40002,45002)
                    whi = random.randint(0,2)
                    if whi == 0 and (rep,i[1],i[2]) not in board:
                        neg.append((rep,i[1],i[2]))
                    if whi == 1 and (i[0],rep,i[2]) not in board:
                        neg.append((i[0],rep,i[2]))
                negative_list.extend(neg)
            #print (count)

            #for k in range(len(word2num_)):
                #if word2num[k] > 40000:
            #    if word2num_[k] < 1:
            #        continue
            #    number = random.randint(5, 11)
            #    i_word_json = self.cooccurance[int(word2num_[k]) - 1]
            #    i_word_co = random.sample(i_word_json, min(number, len(i_word_json)))
            #    for l in range(len(i_word_co)):
            #        word_x.append(int(word2num_[k]))
            #        word_y.append(int(i_word_co[l].encode('utf-8')))

            total_list.append([positive_list, negative_list,self.doc_doc[d_str]])
        print (len(total_list))

        for k in range(len(total_list)):
            total_list[k] = al_ready[k] + total_list[k]
        np.random.shuffle(total_list)
        return total_list

    def get_batch(self,batch_size,data):#data = total_list

        all_batch = data

        batch_index = self.start + batch_size

        bac_li=[]
        sent_li=[]
        sent_entity_li=[]
        sent_len_li=[]
        #word_x_li=[]
        #word_y_li=[]
        ph_li=[]
        pt_li=[]
        pr_li=[]
        nh_li=[]
        nt_li=[]
        nr_li=[]
        ground_truth_li = []
        doc_pr_li = []
        for i in all_batch[self.start:batch_index]:
            po = []
            bac, sent, sent_entity, sent_len, ground_truth,positive_list,negative_list,doc_present = i

            bac_li.append(bac)
            sent_li.append(sent)
            sent_entity_li.append(sent_entity)
            sent_len_li.append(sent_len)
            doc_pr_li.append(doc_present)
            #word_x_li.extend(i for i in word_x)
            #word_y_li.extend(i for i in word_y)
            for x in positive_list:
                po.append(x)
                po.append(x)
                po.append(x)
        #po.append(x)
        #po.append(x)

            ph_li.extend(i[0] for i in po)
            pt_li.extend(i[1] for i in po)
            pr_li.extend(i[2] for i in po)
            nh_li.extend(i[0] for i in negative_list)
            nt_li.extend(i[1] for i in negative_list)
            nr_li.extend(i[2] for i in negative_list)

            ground_truth_li.append(ground_truth)

        self.start = batch_index

        ph_li = asarray_int32(ph_li)
        pt_li= asarray_int32(pt_li)
        pr_li = asarray_int32(pr_li)
        nh_li = asarray_int32(nh_li)
        nt_li = asarray_int32(nt_li)
        nr_li = asarray_int32(nr_li)
        #word_x_li = asarray_int32(word_x_li)
        #word_y_li = asarray_int32(word_y_li).reshape(-1, 1)
        sent_li = asarray_int32(sent_li)
        sent_entity_li = asarray_int32(sent_entity_li)
        bac_li = asarray_int32(bac_li)
        sent_len_li = asarray_int32(sent_len_li)
        doc_pr_li = asarray_int32(doc_pr_li)

        return bac_li,sent_li,sent_entity_li,sent_len_li,ph_li,pt_li,pr_li,nh_li,nt_li,nr_li,ground_truth_li,doc_pr_li



