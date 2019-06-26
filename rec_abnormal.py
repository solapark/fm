import random
import os
import numpy as np
import tensorflow as tf
from W_B_manager import W_B_MANAGER
from context_fm import CONTEXT_FM
from data_loader import DATA_LOADER
from common import *
from parser import save_pickle, load_pickle
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def save_user_and_rr(enc_path, enc_train_name,train_dic_path, item_property_binary_enc_path, model_path, W_user_id_path, B_user_id_path, W_item_id_path, B_item_id_path, w_dim, lr, loss_type, user_path, rr_path): 
    w_b_manager = W_B_MANAGER(W_user_id_path, B_user_id_path, W_item_id_path, B_item_id_path)
    W_B = w_b_manager.load()
    train_loader = DATA_LOADER(enc_path, enc_train_name, train_dic_path, item_property_binary_enc_path, 1, W_B, is_train=False)

    whole_user = []
    whole_rr = []

    tf.reset_default_graph()
    with tf.Session() as session:
        fm = CONTEXT_FM(w_dim, lr, loss_type, item_size = 25)
        #fm = FM(sess_dim, item_dim, w_dim, lr, gpu_id, loss_type)
        saver = tf.train.Saver()
        saver.restore(session, model_path)    
        for i in range(train_loader.impression_dic_size) :
            filter_idx, platform_idx, device_idx, W_user_id, B_user_id, price, dp_order, interaction, W_item_id, B_item_id, item_property_binary, _, _, _, click_idx, _ = train_loader.get_batch()

            if(click_idx == -1):
                print('click not in impression')
                continue
            feed_dict_test = {fm.filter_idx : filter_idx, \
                fm.platform_idx : platform_idx,\
                fm.device_idx : device_idx,\
                fm.interaction : interaction,\
                fm.price : price,\
                fm.order : dp_order,\
                fm.W_user_id_ph : W_user_id,\
                fm.W_item_id_ph : W_item_id,\
                fm.B_user_id_ph : B_user_id,\
                fm.B_item_id_ph : B_item_id,\
                fm.item_property_binary : item_property_binary,\
                fm.click_idx : click_idx\
            }

            cur_recip_rank = session.run(fm.reciprocal_rank, feed_dict=feed_dict_test)

            whole_user.append(train_loader.cur_dic["user_id_idx"])
            whole_rr.append(cur_recip_rank)
            
            if (i %1000 == 0):
                print(i+1 , '/',  train_loader.impression_dic_size, 'Done')

        save_pickle(user_path, whole_user)
        save_pickle(rr_path, whole_rr)

def draw_train_test_relation(train_user, train_rr, val_user, val_rr):
    common_user = list(set(train_user).intersection(set(val_user)))
    common_pick = np.array([[0,0]])
    for user in common_user :
        t_pick = np.round(train_pick[train_user == user])
        v_pick = np.round(val_pick[val_user == user])

        t_pick_size = t_pick.shape[0]
        v_pick_size = v_pick.shape[0]
        t_pick_rep = np.expand_dims(np.tile(t_pick, v_pick_size), axis = 1)
        v_pick_rep = np.expand_dims(np.tile(v_pick, t_pick_size), axis = 1)

        conc = np.concatenate((t_pick_rep, v_pick_rep), axis = 1 ) 
        common_pick = np.concatenate((common_pick, conc), axis = 0)
    common_pick = common_pick[1:]

    common_train_pick = common_pick[:, 0]
    common_val_pick = common_pick[:, 1]
    plt.title("User's Choice in Traing Set and Test Set")
    plt.xlabel('click out item in training set')
    plt.ylabel('click out item in test set')
    plt.hist2d(common_train_pick, common_val_pick, (50, 50), cmap = plt.cm.jet)
    plt.colorbar()
    plt.show()

def draw_bf_af_hist(val_rr_org, val_rr_rev):
    x_label = range(1,26)
    kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=25)
    plt.title("click out item in suggestion")
    plt.xlabel('click out item')
    plt.hist(np.round(1/val_rr_org), **kwargs, color = 'r', label='before reordering')
    plt.hist(np.round(1/val_rr_rev), **kwargs, color = 'b', label = 'after reordering')
    plt.xticks(x_label)
    plt.legend()
    plt.show()

def draw_train_hist(train_pick):
    plt.title("click out item in suggestion")
    plt.xlabel('click out item')
    x_label = range(1,26)
    plt.hist(train_pick, bins=25, normed=True)
    plt.xticks(x_label)
    plt.show()

if __name__ == "__main__" :
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    train_enc_path ='/home/sapark/class/ml/final/data/pkl/train_val_encoding.hdf5'
    test_enc_path ='/home/sapark/class/ml/final/data/pkl/train_val_encoding.hdf5'
 
    enc_train_name = 'train_enc'
    train_dic_path = '/hdd/sap/ml/final/data/pkl/train_dic.pkl'
    enc_test_name = 'val_enc'
    test_dic_path = '/hdd/sap/ml/final/data/pkl/val_dic.pkl'

    item_property_binary_enc_path = '/hdd/sap/ml/final/data/pkl/feature_list/item_proprerty_encoding.npy'

    W_user_id_path = '/hdd/sap/ml/final/data/WB/W_user_id_norm_pd_6110000.npy'
    B_user_id_path = '/hdd/sap/ml/final/data/WB/B_user_id_norm_pd_6110000.npy'
    W_item_id_path = '/hdd/sap/ml/final/data/WB/W_item_id_norm_pd_6110000.npy'
    B_item_id_path = '/hdd/sap/ml/final/data/WB/B_item_id_norm_pd_6110000.npy'
    #checkpoint_path=tf.train.latest_checkpoint(model_path)
    model_path="/hdd/sap/ml/final/model_norm_pd-6110000"

    w_dim =10 
    lr = 1
    loss_type = 'TOP1'

    train_user_path = './analysis/train_user.pkl'
    train_rr_path = './analysis/train_rr.pkl'
    test_user_path = './analysis/test_user.pkl'
    test_rr_path = './analysis/test_rr.pkl'

    #0. train FM model  with training set
    # train.py

    #1. given FM model trained with training set,
    #1) get user's id and rr in training set
    save_user_and_rr(train_enc_path, enc_train_name,train_dic_path, item_property_binary_enc_path, model_path,  W_user_id_path, B_user_id_path, W_item_id_path, B_item_id_path, w_dim, lr, loss_type, train_user_path, train_rr_path)
    #2) get user's id and rr in test set
    save_user_and_rr(test_enc_path, enc_test_name,test_dic_path, item_property_binary_enc_path, model_path,  W_user_id_path, B_user_id_path, W_item_id_path, B_item_id_path, w_dim, lr, loss_type, test_user_path, test_rr_path)

    train_user = np.array(load_pickle(train_user_path))
    train_rr = np.array(load_pickle(train_rr_path))
    train_mrr = np.mean(train_rr)
    train_pick = 1 / train_rr

    val_user = np.array(load_pickle(test_user_path))
    val_rr = np.array(load_pickle(test_rr_path))
    val_mrr = np.mean(val_rr)
    val_pick = 1 / val_rr

    print('train_mrr', train_mrr)
    print('val_mrr', val_mrr)

    #2. if train_rr < 1/13, reverse order of suggestion.
    train_rr_org = np.array([])
    val_rr_org = np.array([])
    val_rr_rev = np.array([])
    common_user = list(set(train_user).intersection(set(val_user)))
    for user in common_user :
        t_pick = np.round(train_pick[train_user == user])
        t_pick_mean = np.mean(t_pick)
        v_pick_org = np.round(val_pick[val_user == user])

        train_rr_org = np.append(train_rr_org, 1/t_pick)
        val_rr_org = np.append(val_rr_org, 1/v_pick_org)
        #if(t_pick_mean > 20) : 
        if(t_pick_mean < 5) : 
            #v_pick = 26 - v_pick_org
            v_pick = np.clip(v_pick_org-1, 1, 25)
        else :
            v_pick = v_pick_org
    
        val_rr_rev = np.append(val_rr_rev, 1/v_pick)
     
    print('val_mrr', np.mean(val_rr_org), 'val_revised_mrr', np.mean(val_rr_rev))

    draw_train_hist(train_pick)
    draw_bf_af_hist(val_rr_org, val_rr_rev)
    draw_train_test_relation(train_user, train_rr, val_user, val_rr)
