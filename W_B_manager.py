from common import *
from scipy.stats import truncnorm
import numpy as np

class W_B_MANAGER:
    def __init__(self, W_user_id_path, B_user_id_path, W_item_id_path, B_item_id_path):
        self.W_user_id_path = W_user_id_path
        self.B_user_id_path = B_user_id_path
        self.W_item_id_path = W_item_id_path
        self.B_item_id_path = B_item_id_path

        self.W_user_id_path_name= self.split_str(W_user_id_path, '.')[0]
        self.B_user_id_path_name = self.split_str(B_user_id_path, '.')[0]
        self.W_item_id_path_name = self.split_str(W_item_id_path, '.')[0]
        self.B_item_id_path_name = self.split_str(B_item_id_path, '.')[0]

        self.user_size = data_size.USER_ID.value
        self.item_size = data_size.ITEM_ID.value

    def new(self, W_dim) :
        W_user_id = np.ones([self.user_size, W_dim])
        B_user_id = np.ones([self.user_size, ])
        W_item_id = np.ones([self.item_size, W_dim])
        B_item_id = np.ones([self.item_size, ])
        '''
        truc = truncnorm(-2, 2, loc = 0.0, scale = 1.0)
        W_user_id = truc.rvs(self.user_size * W_dim).reshape([self.user_size, W_dim])
        B_user_id = truc.rvs(self.user_size)
        W_item_id = truc.rvs(self.item_size * W_dim).reshape([self.item_size, W_dim])
        B_item_id = truc.rvs(self.item_size)
    '''
        return W_user_id, B_user_id, W_item_id, B_item_id

    def load(self) :
        W_user_id = np.load(self.W_user_id_path)
        B_user_id = np.load(self.B_user_id_path)
        W_item_id = np.load(self.W_item_id_path)
        B_item_id = np.load(self.B_item_id_path)
        return W_user_id, B_user_id, W_item_id, B_item_id

    def split_str(self, target, token) :
        return target.split(token)

    def add_step_to_name(self, step, name):
        dst = name+'_'+str(step)
        return dst

    def get_path_with_step(self, step) :
         W_user_id_path_name = self.add_step_to_name(step, self.W_user_id_path_name)
         B_user_id_path_name = self.add_step_to_name(step, self.B_user_id_path_name)
         W_item_id_path_name = self.add_step_to_name(step, self.W_item_id_path_name)
         B_item_id_path_name = self.add_step_to_name(step, self.B_item_id_path_name)

         return W_user_id_path_name, B_user_id_path_name, W_item_id_path_name, B_item_id_path_name

    def save(self, W_user_id, B_user_id, W_item_id, B_item_id, step) :
         W_user_id_path_name, B_user_id_path_name, W_item_id_path_name, B_item_id_path_name = self.get_path_with_step(step)
         np.save(W_user_id_path_name, W_user_id)
         np.save(B_user_id_path_name, B_user_id)
         np.save(W_item_id_path_name, W_item_id)
         np.save(B_item_id_path_name, B_item_id)

if __name__ == "__main__" :
    W_user_id_path = '/hdd/sap/ml/final/data/WB/W_user_id.npy'
    B_user_id_path = '/hdd/sap/ml/final/data/WB/B_user_id.npy'
    W_item_id_path = '/hdd/sap/ml/final/data/WB/W_item_id.npy'
    B_item_id_path = '/hdd/sap/ml/final/data/WB/B_item_id.npy'
    w_dim =10 

    w_b_manager = W_B_MANAGER(W_user_id_path, B_user_id_path, W_item_id_path, B_item_id_path)
    W_user_id, B_user_id, W_item_id, B_item_id = w_b_manager.new(w_dim)
    print(W_user_id[0, 0:5], B_user_id[0], W_item_id[0, 0:5], B_item_id[0])
    w_b_manager.save(W_user_id, B_user_id, W_item_id, B_item_id, 10)

    W_user_id_path = '/hdd/sap/ml/final/data/WB/W_user_id_10.npy'
    B_user_id_path = '/hdd/sap/ml/final/data/WB/B_user_id_10.npy'
    W_item_id_path = '/hdd/sap/ml/final/data/WB/W_item_id_10.npy'
    B_item_id_path = '/hdd/sap/ml/final/data/WB/B_item_id_10.npy'

    w_b_manager = W_B_MANAGER(W_user_id_path, B_user_id_path, W_item_id_path, B_item_id_path)
    W_user_id, B_user_id, W_item_id, B_item_id = w_b_manager.load()

    print(W_user_id[0, 0:5], B_user_id[0], W_item_id[0, 0:5], B_item_id[0])
