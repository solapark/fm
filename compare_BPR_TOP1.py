import matplotlib.pyplot as plt
import random
import os
import numpy as np
import tensorflow as tf
from fm import FM
from data_loader import DATA_LOADER
from common import *

def save_loss_mrr(enc_path, enc_train_name,train_dic_path, sess_dim, item_dim, w_dim, lr, gpu_id, loss_path, mrr_path, loss_type):
    tf.reset_default_graph()

    train_loader = DATA_LOADER(enc_path, enc_train_name, train_dic_path, train_batch_size, is_train=True)
    val_loader = DATA_LOADER(enc_path, enc_val_name, val_dic_path, test_batch_size, is_train=False)
    fm = FM(sess_dim, item_dim, w_dim, lr, gpu_id, loss_type)

    loss_array = np.array([[0.0, 0.0]])
    mrr_array = np.array([[0.0, 0.0]])
    loss = 0
    loss_cnt = 0
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
    
        for step in range(num_steps+1):
            sess, item, y, dic = train_loader.get_batch()
            feed_dict_train = {fm.session: sess,
                               fm.item: item,
                               fm.y: y}
            cur_loss = session.run(fm.loss, feed_dict=feed_dict_train)
            if not np.isnan(float(cur_loss)) and not np.isinf(float(cur_loss)): 
                loss = loss + cur_loss 
                loss_cnt +=1
            else :
                print('error', step, cur_loss)
                continue

            if (step % 100 == 0):
                print("@ iter %d loss : %f" %(step, loss/loss_cnt))
                loss_array = np.append(loss_array, np.array([[step, loss/loss_cnt]]), axis = 0)
                loss = 0
                loss_cnt = 0

            if (step % 1000 == 0):
                recip_rank = 0
                recip_cnt = 0
                for i in range(val_loader.dic_size):
                #for i in range(500):
                    sess, item, _, dic = val_loader.get_batch()
                    if(dic[0]["click"] == -1):
                        print('click not in impression', dic[0])
                        continue
                    feed_dict_test = {fm.session: sess,
                                      fm.item: item,
                                      fm.click_idx: dic[0]["click"]}
                    cur_recip_rank = session.run(fm.reciprocal_rank, feed_dict=feed_dict_test)
                    if not np.isinf(float(cur_recip_rank)) :
                        recip_rank += cur_recip_rank
                        recip_cnt += 1
                print("@@@@@@@@@@@@@@@ iter %d mrr: %f" %(step, recip_rank/recip_cnt))
                mrr_array = np.append(mrr_array, np.array([[step, recip_rank/recip_cnt]]), axis = 0)
        
            session.run(fm.optimize, feed_dict=feed_dict_train)

    np.save(loss_path, loss_array[1:])
    np.save(mrr_path, mrr_array[1:])

def draw_2mrr(ax, title, BPR_mrr, TOP1_mrr):
    ax.set_title(title)
    ax.set_xlabel('iteration')
    ax.set_ylabel('mrr')
    ax.plot(TOP1_mrr[:, 0], TOP1_mrr[:, 1], color='r', marker = 'o', label = 'TOP1_mrr')
    ax.plot(BPR_mrr[:, 0], BPR_mrr[:, 1], color='b', marker = 'o', label = 'BPR_mrr')
    ax.legend()

def draw_loss_and_mrr(loss_ax, title, loss, mrr):
    loss_ax.set_title(title)
    loss_ax.set_xlabel('iteration')
    loss_ax.set_ylabel('loss')
    mrr_ax = loss_ax.twinx()
    loss_ax.plot(loss[:, 0], loss[:, 1], color='r', marker = 'o', label = 'loss')
    mrr_ax.plot(mrr[:, 0], mrr[:, 1], color='b', marker = 'o', label = 'mrr')
    mrr_ax.set_ylabel('mrr')
    
    loss_ax.legend(loc = 'upper left')
    #loss_ax.legend(loc = 'upper right')
    mrr_ax.legend(loc = 'upper right')

os.environ["CUDA_VISIBLE_DEVICES"]='2'
enc_path ='/data1/sap/ml/final/data/pkl/train_val_encodingi_10k.hdf5'
 
enc_train_name = 'train_enc'
enc_val_name = 'val_enc'
train_dic_path = '/data1/sap/ml/final/data/pkl/train_dic_10k.pkl'
val_dic_path = '/data1/sap/ml/final/data/pkl/val_dic_10k.pkl'

loss_BPR_path = '/data1/sap/ml/final/data/pkl/loss_BPR.npy'
mrr_BPR_path = '/data1/sap/ml/final/data/pkl/mrr_BPR.npy'
loss_TOP1_path = '/data1/sap/ml/final/data/pkl/loss_TOP1.npy'
mrr_TOP1_path = '/data1/sap/ml/final/data/pkl/mrr_TOP1.npy'

num_steps = 5000
#num_steps = 1
train_batch_size =10
test_batch_size = 1
sess_dim = feature_size.SESSION.value 
item_dim = feature_size.ITEM.value
w_dim =10 
lr = 0.1
gpu_id = ['/gpu:0']
random.seed(0)

#save_loss_mrr(enc_path, enc_train_name,train_dic_path, sess_dim, item_dim, w_dim, lr, gpu_id, loss_BPR_path, mrr_BPR_path, loss_type='BPR')
#save_loss_mrr(enc_path, enc_train_name,train_dic_path, sess_dim, item_dim, w_dim, lr, gpu_id, loss_TOP1_path, mrr_TOP1_path, loss_type='TOP1')

loss_TOP1 = np.load(loss_TOP1_path)
mrr_TOP1 = np.load(mrr_TOP1_path)
loss_BPR = np.load(loss_BPR_path)
mrr_BPR = np.load(mrr_BPR_path)


BRT_ax = plt.subplot(3, 1,1)
TOP1_ax = plt.subplot(3, 1,2)
mrr_comp_ax = plt.subplot(3, 1,3)
draw_loss_and_mrr(BRT_ax, 'BPR', loss_BPR, mrr_BPR)
draw_loss_and_mrr(TOP1_ax, 'TOP1', loss_TOP1, mrr_TOP1)
draw_2mrr(mrr_comp_ax, 'BPR vs TOP1', mrr_BPR, mrr_TOP1)
plt.tight_layout()
plt.show()

