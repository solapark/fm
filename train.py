import random
import os
import numpy as np
import tensorflow as tf
from context_fm import CONTEXT_FM
from data_loader import DATA_LOADER
from W_B_manager import W_B_MANAGER
from common import *

W_user_id_path = '/data1/sap/ml/final/data/WB/W_user_id_norm_pd.npy'
B_user_id_path = '/data1/sap/ml/final/data/WB/B_user_id_norm_pd.npy'
W_item_id_path = '/data1/sap/ml/final/data/WB/W_item_id_norm_pd.npy'
B_item_id_path = '/data1/sap/ml/final/data/WB/B_item_id_norm_pd.npy'

model_path = model_path + '_norm_pd'

'''
W_user_id_path = '/data1/sap/ml/final/data/WB/W_user_id_350000.npy'
B_user_id_path = '/data1/sap/ml/final/data/WB/B_user_id_350000.npy'
W_item_id_path = '/data1/sap/ml/final/data/WB/W_item_id_350000.npy'
B_item_id_path = '/data1/sap/ml/final/data/WB/B_item_id_350000.npy'
'''

os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
epoch = 100
num_steps = epoch*data_size.TRAIN_SESSION.value
batch_size =1
w_dim =50 
#lr = 0.01 
lr = 1 
#loss_type = 'BPR'
loss_type = 'TOP1'

w_b_manager = W_B_MANAGER(W_user_id_path, B_user_id_path, W_item_id_path, B_item_id_path)
W_B = w_b_manager.new(w_dim)
#W_B = w_b_manager.load()
train_loader = DATA_LOADER(impression_enc_path, impression_enc_train_name, impression_dic_train_path, item_property_binary_enc_path, batch_size, W_B, is_train=True)
val_loader = DATA_LOADER(impression_enc_path, impression_enc_val_name, impression_dic_val_path, item_property_binary_enc_path, batch_size, W_B, is_train=False)
fm = CONTEXT_FM(w_dim, lr, loss_type, item_size = 25)

loss_array = np.array([[0.0, 0.0]])
mrr_array = np.array([[0.0, 0.0]])
loss = 0
loss_cnt = 0
with tf.Session() as session:
    saver = tf.train.Saver(max_to_keep=600)
    session.run(tf.global_variables_initializer())
    #saver.restore(session, "/data1/sap/ml/final/model/model-350000")
    #saver.restore(session, tf.train.latest_checkpoint(model_path))
    for step in range(num_steps):
        filter_idx, platform_idx, device_idx, W_user_id, B_user_id, price, dp_order, interaction, W_item_id, B_item_id, item_property_binary, y, _, _, _, _ = train_loader.get_batch()
        #dp_order = dp_order/24
        price_mean = np.mean(price)
        price_std = max(np.std(price), 1)
        price =  - ( (price - price_mean) / price_std )
        '''
        if len(price) < 25 : 
            #price = np.append(price, np.array( [max(price)] * (25-len(price) ))) 
            price = np.append(price, np.array( [1000] * (25-len(price) ))) 
            #print(price)
        '''
        #interaction = interaction / np.clip(np.sum(interaction), 1, 10000) 
        interaction = interaction > 0
        item_property_binary = item_property_binary / np.clip(np.sum(item_property_binary), 1, 10000)
        filter_idx = filter_idx / np.clip(np.sum(filter_idx), 1, 10000)
        feed_dict_train = {fm.filter_idx : filter_idx, \
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
                    fm.y : y\
                }

        #_, cur_loss = session.run([fm.assign_var, fm.loss], feed_dict=feed_dict_train)
        _, cur_loss = session.run([fm.assign_var, fm.temp_loss], feed_dict=feed_dict_train)
        #cur_loss = session.run(fm.temp_loss, feed_dict=feed_dict_train)
        if np.isnan(float(cur_loss[-1])) or  np.isinf(float(cur_loss[-1])):
            print('step', step)
            print('price', price)
            print(session.run(fm.logits, feed_dict = feed_dict_train))
            #print(cur_loss)
        loss = loss + cur_loss[-1] 
        loss_cnt +=1

        if (step % 1000 == 0):
            print("@ iter %d loss : %f" %(step, loss/loss_cnt))
            loss = 0
            loss_cnt = 0

        if (step % 10000 == 0):
            new_user_W, new_user_B, new_item_W, new_item_B = train_loader.get_WB()
            val_loader.change_WB(new_user_W, new_user_B, new_item_W, new_item_B)
            recip_rank = 0
            recip_cnt = 0
            for i in range(val_loader.impression_dic_size):
                filter_idx, platform_idx, device_idx, W_user_id, B_user_id, price, dp_order, interaction, W_item_id, B_item_id, item_property_binary, _, _, _, click_idx, _ = val_loader.get_batch()

                #dp_order = dp_order/24
                price_mean = np.mean(price)
                price_std = max(np.std(price), 1)
                price =  - ( (price - price_mean) / price_std )
                '''
                if len(price) < 25 : 
                    #price = np.append(price, np.array( [max(price)] * (25-len(price) ))) 
                    price = np.append(price, np.array( [1000] * (25-len(price) ))) 
                    #print(price)
 
                '''
                #interaction = interaction / np.clip(np.sum(interaction), 1, 10000) 
                interaction = interaction > 0
                #interaction = interaction * 0
                item_property_binary = item_property_binary / np.clip(np.sum(item_property_binary), 1, 10000)
                filter_idx = filter_idx / np.clip(np.sum(filter_idx), 1, 10000)
                if(click_idx == -1):
                    #print('click not in impression')
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

                #cur_recip_rank = session.run(fm.reciprocal_rank, feed_dict=feed_dict_test)
                _, cur_recip_rank = session.run([fm.assign_var, fm.reciprocal_rank], feed_dict=feed_dict_test)
                recip_rank += cur_recip_rank
                recip_cnt += 1
            print("@@@@@@@@@@@@@@@ iter %d mrr: %f" %(step, recip_rank/recip_cnt))
            #saver.save(session, os.path.join(model_path, 'model'), global_step=step) 
            saver.save(session, model_path, global_step=step) 
            #print(tf.train.latest_checkpoint(os.path.join(model_path, 'model')))
            w_b_manager.save(new_user_W, new_user_B, new_item_W, new_item_B, step)
            #save_loss_mrr()
        
        #if(step <20000 ) : 
        if(int( step /5000) % 2 == 0) : 
            #print('fm.optimize')
            session.run(fm.optimize, feed_dict=feed_dict_train)
        else : 
            #print('fm.optimize_user')
            session.run(fm.optimize_user, feed_dict=feed_dict_train)
        #session.run(fm.optimize, feed_dict=feed_dict_train)
        new_user_W, new_user_B, new_item_W, new_item_B = session.run([fm.get_W_B_user_id_item_id()])[0]
        '''
        print('old_user_W', W_user_id)
        print('new_user_W', new_user_W)
        print('new_user_B', new_user_B)
        '''
        train_loader.update_WB(new_user_W, new_user_B, new_item_W, new_item_B)
