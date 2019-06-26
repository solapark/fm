from common import *
from context_fm import CONTEXT_FM
from W_B_manager import W_B_MANAGER
from data_loader import DATA_LOADER
from parser import load_pickle
import csv
import os
import tensorflow as tf
import numpy as np

class SUBMISSION:
    def __init__(self, file_path, is_val=False):
        self.csv = open(file_path, 'w')
        header = submission_header
        if is_val : 
            header.insert(2, 'pick')
            header.insert(3, 'click')
        self.writer = csv.DictWriter(self.csv, fieldnames = header)

        self.writer.writeheader()
 
    def write(self, user_id, session_id, timestamp, step, item_reccomendations_list) :
        self.writer.writerow({'user_id' : user_id, 'session_id' : session_id, 'timestamp' : timestamp, 'step' : step, 'item_recommendations' : item_reccomendations_list})

    def write_is_val(self, user_id, session_id, timestamp, step, item_reccomendations_list, click, pick) :
        self.writer.writerow({'user_id' : user_id, 'session_id' : session_id, 'timestamp' : timestamp, 'step' : step, 'click' : click, 'item_recommendations' : item_reccomendations_list, 'pick' : pick})

    def close(self) :
        self.csv.close()

if __name__ == "__main__" :
    W_user_id_path = '/hdd/sap/ml/final/data/WB/W_user_id_norm_pd_7810000.npy'
    B_user_id_path = '/hdd/sap/ml/final/data/WB/B_user_id_norm_pd_7810000.npy'
    W_item_id_path = '/hdd/sap/ml/final/data/WB/W_item_id_norm_pd_7810000.npy'
    B_item_id_path = '/hdd/sap/ml/final/data/WB/B_item_id_norm_pd_7810000.npy'
    #checkpoint_path=tf.train.latest_checkpoint(model_path)
    checkpoint_path="/hdd/sap/ml/final/model_norm_pd-7810000"
 
    '''
    W_user_id_path = '/hdd/sap/ml/final/data/WB/W_user_id_norm_pd_400000.npy'
    B_user_id_path = '/hdd/sap/ml/final/data/WB/B_user_id_norm_pd_400000.npy'
    W_item_id_path = '/hdd/sap/ml/final/data/WB/W_item_id_norm_pd_400000.npy'
    B_item_id_path = '/hdd/sap/ml/final/data/WB/B_item_id_norm_pd_400000.npy'
    checkpoint_path="/hdd/sap/ml/final/model_norm_pd-400000"
    '''
 
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    w_dim =10 
    lr = 1
    batch_size = 1
    loss_type = 'TOP1'
    
    session_id_list = load_pickle(session_id_list_path)
    user_id_list = load_pickle(user_id_list_path)

    w_b_manager = W_B_MANAGER(W_user_id_path, B_user_id_path, W_item_id_path, B_item_id_path)
    W_B = w_b_manager.load()
    test_loader = DATA_LOADER(test_enc_path, enc_test_name, test_dic_path, item_property_binary_enc_path, batch_size, W_B, is_train=False)
    test_loader = DATA_LOADER(impression_enc_path, impression_enc_val_name, impression_dic_val_path, item_property_binary_enc_path, batch_size, W_B, is_train=False)
    sumission_csv_path = '/home/sapark/class/ml/final/submission/val_sub7.csv' 
    #sumission_csv_path = '/home/sapark/class/ml/final/submission/sub7.csv' 
    #submission = SUBMISSION(sumission_csv_path)
    submission = SUBMISSION(sumission_csv_path, is_val = True)
    fm = CONTEXT_FM(w_dim, lr, loss_type, item_size = 25)
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_path)    
        for i in range(test_loader.impression_dic_size) :
            filter_idx, platform_idx, device_idx, W_user_id, B_user_id, price, dp_order, interaction, W_item_id, B_item_id, item_property_binary, _, _, _, _, impressions = test_loader.get_batch()
            price_mean = np.mean(price)
            price_std = max(np.std(price), 1)
            price =  - ( (price - price_mean) / price_std )
            interaction = interaction > 0
            item_property_binary = item_property_binary / np.clip(np.sum(item_property_binary), 1, 10000)
            filter_idx = filter_idx / np.clip(np.sum(filter_idx), 1, 10000)

            impressions = impressions.split('|')
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
                fm.impression : impressions\
            }

            cur_recip_rank = session.run(fm.prediction, feed_dict=feed_dict_test)
            cur_recip_rank = cur_recip_rank.astype('U13').tolist()
            space_sep_rank = ' '.join(cur_recip_rank)
            user_id = user_id_list[test_loader.cur_dic["user_id_idx"]]
            session_id = session_id_list[test_loader.cur_dic["session_id_idx"]]
            timestamp = test_loader.cur_dic["timestamp"]
            step = test_loader.cur_dic["step"]
            click = impressions[test_loader.cur_dic["click"]] if test_loader.cur_dic["click"] != -1 else -1
            pick = cur_recip_rank.index(click) + 1 if click != -1 else -1
            #submission.write(user_id, session_id, timestamp, step, space_sep_rank)
            submission.write_is_val(user_id, session_id, timestamp, step, space_sep_rank, click, pick)
            
            if (i %1000 == 0):
                print(i+1 , '/',  test_loader.impression_dic_size, 'Done')

    submission.close()
