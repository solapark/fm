import time
import numpy as np
import random
from parser import load_hdf, load_pickle, decode_dic, decode_batch
from common import *
from W_B_manager import W_B_MANAGER

random.seed(0)
class DATA_LOADER:
    def __init__(self, impression_enc_path, impression_enc_name, impression_dic_path, item_property_binary_enc_path, batch_size, W_B, is_train):
        self.impression_enc = load_hdf(impression_enc_path, impression_enc_name) 
        self.impression_dic = load_pickle(impression_dic_path)
        self.impression_dic_size = len(self.impression_dic)

        #self.item_property_binary_enc = load_hdf(item_property_binary_enc_path, item_property_binary_enc_name) 
        self.item_property_binary_enc = np.load(item_property_binary_enc_path) 

        self.W_user_id, self.B_user_id, self.W_item_id, self.B_item_id = W_B
        self.W_dim = self.W_item_id.shape[1]
        self.is_train = is_train
        self.batch_size = batch_size
        self.last_batch_idx = 0

        if(self.is_train) :
            self.shuffle_batch()

        self.cur_dic = 0
        self.cur_click_idx = 0
        self.cur_filter_idx = 0
        self.cur_platform_idx = 0
        self.cur_country_idx = 0
        self.cur_device = 0
        self.cur_click_list = 0
        self.cur_user_id = 0
        self.cur_W_user_id = 0
        self.cur_B_user_id = 0
        self.cur_enc = 0
        self.cur_price = 0
        self.cur_dp_order = 0
        self.cur_interaction = 0
        self.cur_item_id = 0
        self.cur_num_item = 0 
        self.cur_item_property_binary = 0
        self.cur_y = 0
        self.impressions = 0

        self.cur_W_item_id = np.array([[0.0] * self.W_dim ] * 25)
        self.cur_B_item_id = np.array([0.0] * 25)

   
    def shuffle_batch(self) :
        random.shuffle(self.impression_dic)

    def get_batch(self) :
        #a = time.time()
        cur_dic = self.impression_dic[self.last_batch_idx: self.last_batch_idx+self.batch_size]
        #b = time.time()

        self.cur_dic = cur_dic[0]

        self.impressions = self.cur_dic["impressions"]
        #c = time.time()
        self.cur_click_idx = self.cur_dic['click']
        #d = time.time()
        self.cur_filter_idx = self.cur_dic['filter_idx']
        #e = time.time()
        self.cur_platform_idx = self.cur_dic['platform_idx']
        self.cur_country_idx = self.cur_dic['country_idx']
        #f = time.time()
        self.cur_device_idx = self.cur_dic['device_idx']
        #g = time.time()
        self.cur_user_id = self.cur_dic['user_id_idx']
        #h = time.time()
        self.cur_W_user_id = self.W_user_id[self.cur_user_id]
        #i = time.time()
        self.cur_B_user_id = self.B_user_id[self.cur_user_id]

        #j = time.time()
        self.cur_enc = self.impression_enc[self.cur_dic["slice"]]
        #k = time.time()
        self.cur_price = self.cur_enc[:, encoding_idx.PRICE.value]
        #l = time.time()
        self.cur_dp_order = self.cur_enc[:, encoding_idx.DP_ORDER.value]
        #m = time.time()
        self.cur_interaction = self.cur_enc[:, encoding_idx.INTERACTION.value]
        #o = time.time()
        self.cur_item_id = self.cur_enc[:, encoding_idx.ITEM_ID.value]
        #p = time.time()
        self.cur_num_item = len(self.cur_item_id)
       #q = time.time()
        self.cur_click_list = np.zeros(self.cur_num_item, dtype= int)
        self.cur_click_list[self.cur_dic["click_list"]] = 1
        '''
        if(self.cur_num_item != 25) :
            print(self.cur_num_item)
            print('bf')
            print(self.cur_W_item_id[:, 0])
        '''
        self.cur_W_item_id[:self.cur_num_item] = self.W_item_id[self.cur_item_id]
        #r = time.time()
        self.cur_B_item_id[:self.cur_num_item] = self.B_item_id[self.cur_item_id]
        #s = time.time()
        '''
        if(self.cur_num_item != 25) :
            print('af')
            print(self.cur_W_item_id[:, 0])
        '''
        #r = time.time()
        self.cur_B_item_id[:self.cur_num_item] = self.B_item_id[self.cur_item_id]
        #s = time.time()
             
        self.cur_item_property_binary = self.item_property_binary_enc[self.cur_item_id]
        #t = time.time()
        self.cur_y = self.cur_enc[:, encoding_idx.Y.value]
        #u = time.time()

        self.last_batch_idx += self.batch_size
        if(self.last_batch_idx + self.batch_size >= self.impression_dic_size) :
            if(self.is_train) :
                self.shuffle_batch()
            self.last_batch_idx = 0

        return self.cur_filter_idx, self.cur_platform_idx, self.cur_country_idx, self.cur_device_idx, self.cur_click_list, self.cur_W_user_id, self.cur_B_user_id, self.cur_price, self.cur_dp_order, self.cur_interaction, self.cur_W_item_id, self.cur_B_item_id, self.cur_item_property_binary, self.cur_y, self.cur_user_id, self.cur_item_id, self.cur_click_idx, self.impressions
        
    def update_WB(self, new_user_W, new_user_B, new_item_W, new_item_B) :
        self.W_user_id[self.cur_user_id] = new_user_W
        self.B_user_id[self.cur_user_id] = new_user_B
        self.W_item_id[self.cur_item_id] = new_item_W[:self.cur_num_item]
        self.B_item_id[self.cur_item_id] = new_item_B[:self.cur_num_item]

    def change_WB(self, new_user_W, new_user_B, new_item_W, new_item_B):
        self.W_user_id = new_user_W
        self.B_user_id = new_user_B
        self.W_item_id = new_item_W
        self.B_item_id = new_item_B

    def get_WB(self):
        return self.W_user_id, self.B_user_id, self.W_item_id, self.B_item_id

if __name__ == "__main__" :
    impression_enc_path ='/home/sap/class/ml/final/data/pkl/train_val_encoding.hdf5' 
    impression_enc_train_name = 'train_enc'
    impression_enc_val_name = 'val_enc'
    impression_dic_train_path = '/data1/sap/ml/final/data/pkl/train_dic.pkl'
    impression_dic_val_path = '/data1/sap/ml/final/data/pkl/val_dic.pkl'
    item_property_binary_enc_path = '/data1/sap/ml/final/data/pkl/feature_list/item_proprerty_encoding.npy'

    W_user_id_path = '/data1/sap/ml/final/data/pkl/W_user_id.np'
    B_user_id_path = '/data1/sap/ml/final/data/pkl/B_user_id.np'
    W_item_id_path = '/data1/sap/ml/final/data/pkl/W_item_id.np'
    B_item_id_path = '/data1/sap/ml/final/data/pkl/B_item_id.np'
    
    W_dim = 10
    batch_size = 1 

    platform_list_path = '/data1/sap/ml/final/data/pkl/feature_list/platform_list.pkl'
    device_list_path = '/data1/sap/ml/final/data/pkl/feature_list/device_list.pkl'
    filter_list_path = '/data1/sap/ml/final/data/pkl/feature_list/filter_list.pkl'
    user_id_list_path = '/data1/sap/ml/final/data/pkl/feature_list/user_id_list.pkl'
    session_id_list_path = '/data1/sap/ml/final/data/pkl/feature_list/session_id_list.pkl'
    item_id_list_path = '/data1/sap/ml/final/data/pkl/feature_list/item_id_list.pkl'
 
    w_b_manager = W_B_MANAGER(W_user_id_path, B_user_id_path, W_item_id_path, B_item_id_path)
    w_b_manager.new_W_B(W_dim)
    W_B = w_b_manager.get_W_B()
    train_loader = DATA_LOADER(impression_enc_path, impression_enc_train_name, impression_dic_train_path, item_property_binary_enc_path, batch_size, W_B, is_train=True)
    val_loader = DATA_LOADER(impression_enc_path, impression_enc_val_name, impression_dic_val_path, item_property_binary_enc_path,  batch_size, W_B, is_train=False)

    filter_idx, platform_idx, device_idx, W_user_id, B_user_id, price, dp_order, interaction, W_item_id, B_item_id, item_property_binary, y, user_id_idx, item_id_idx, click = train_loader.get_batch()


    print('decode_batch')
    session_dic, impression_dics = decode_batch(filter_idx, platform_idx, device_idx, price, dp_order, interaction, y, user_id_idx, item_id_idx, click, platform_list_path, device_list_path, filter_list_path, user_id_list_path, session_id_list_path, item_id_list_path) 
    print(session_dic)
    for impression_dic in impression_dics:
        print(impression_dic) 
    session_dic, impression_dics = decode_dic(train_loader.cur_dic, impression_enc_path, 'train_enc', platform_list_path, device_list_path, filter_list_path, user_id_list_path, session_id_list_path, item_id_list_path)
    print('decode_dic')
    print(session_dic)
    for impression_dic in impression_dics:
        print(impression_dic) 

    new_user_W = [0]*W_dim
    new_user_B = 0
    new_item_W = [[0]*W_dim]*25
    new_item_B = [0] * 25
    start_time = time.time()
    for i in range(train_loader.impression_dic_size) :
        filter_idx, platform_idx, device_idx, W_user_id, B_user_id, price, dp_order, interaction, W_item_id, B_item_id, item_property_binary, y, user_id_idx, item_id_idx, _ = train_loader.get_batch()
        train_loader.update_WB(new_user_W, new_user_B, new_item_W, new_item_B) 
        if i % 10000 == 0 : 
            print(i, '/', train_loader.impression_dic_size, 'time', time.time() - start_time)
            start_time = time.time()
            
    '''
    print('bf\n',train_loader.W_item_id[item_id_idx])
    new_user_W = [0]*W_dim
    new_user_B = 0
    new_item_W = [[0]*W_dim]*25
    new_item_B = [0] * 25
    w_b_manager.update_W_B(user_id_idx, new_user_W, new_user_B, item_id_idx, new_item_W, new_item_B)
    train_loader.update_WB(w_b_manager.get_W_B())
    print('af\n', train_loader.W_item_id[item_id_idx])
    for i in range(3) : 
        print('***************************')
        filter_idx, platform_idx, device_idx, W_user_id, B_user_id, price, dp_order, interaction, W_item_id, B_item_id, item_property_binary, y = train_loader.get_batch()
        print(filter_idx, platform_idx, device_idx, W_user_id, B_user_id, price, dp_order, interaction, W_item_id, B_item_id, item_property_binary, y)
        filter_idx, platform_idx, device_idx, W_user_id, B_user_id, price, dp_order, interaction, W_item_id, B_item_id, item_property_binary, y = val_loader.get_batch()

        print(filter_idx, platform_idx, device_idx, W_user_id, B_user_id, price, dp_order, interaction, W_item_id, B_item_id, item_property_binary, y)
    
    '''
    print('finish')
