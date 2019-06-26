from parser import load_hdf
import numpy as np
import h5py

def hdf2np(hdf_path, data_name, np_path) :
    data = load_hdf(hdf_path, data_name)
    a = np.empty(shape = data.shape)
    for i in range(data.shape[0]):
        a[i] = data[i]
        if i%10000 == 0 : print(i+1, '/',  data.shape[0])
    np.save(np_path, a)

if __name__ == "__main__" :
    item_property_binary_enc_path = '/hdd/sap/ml/final/data/pkl/feature_list/item_proprerty_encoding.hdf5'
    item_property_binary_enc_name = 'item_property_binary'
    item_property_binary_np_path = '/hdd/sap/ml/final/data/pkl/feature_list/item_proprerty_encoding'
    #hdf2np(item_property_binary_enc_path, item_property_binary_enc_name, item_property_binary_np_path)
    '''
    a = np.load(item_property_binary_np_path)
    print(a[0])
    '''
