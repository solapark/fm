import os
import h5py
import glob
import numpy as np
import csv
import pickle
import enum
import time
from scipy.sparse import csr_matrix
import bisect
from common import *

def add_str_to_path(path, token) :
    name = path.split('.')[0]
    exp = path.split('.')[1]
    new_path = name + token + '.' + exp
    return new_path

def make_item_id_list(csv_file, header_name):
    f = open(csv_file, 'r')
    lines = csv.reader(f)
    header = next(lines, None)
    print('header', header)
    target_idx = header.index(header_name)
    target_list =list() 
    for i, line in enumerate(lines) : 
        target = line[target_idx]
        if(target):
            target_list.append(int(target))
    
        if(i % 100000 == 0) :
            print(i+1, 'DONE') 
    f.close()

    pickle_name = '/hdd/sap/ml/final/data/' + header_name + '.pkl'
    save_pickle(pickle_name, target_list)

def get_feature_set(csv_file_name, header_name, action_type_token):
    f = open(csv_file_name, 'r')
    lines = csv.reader(f)
    header = next(lines, None)
    #print('header', header)
    target_idx = header.index(header_name)
    if(action_type_token) :
        action_type_idx = header_idx.ACTION_TYPE.value
    else :
        action_type_idx = 0 
    target_set =set() 
    for i, line in enumerate(lines) : 
        target = line[target_idx]
        action_type = line[action_type_idx]
        if(target):
            if(action_type_token and action_type != action_type_token): continue
            target_tokens = target.split('|')
            for target_token in target_tokens :
                target_set.add(target_token)
    
        if(i % 100000 == 0) :
            print(i+1, 'DONE') 

    f.close()
    return target_set 

def make_feature_list(train_csv, test_csv, item_metadata_csv, header_name, save_path, is_item_metadata, action_type_token = None):
    target_set = set()
    if(is_item_metadata) :
        print('item_metatdata parsing....')
        meta_set =get_feature_set(item_metadata_csv, header_name, action_type)
        target_set = meta_set
    else : 
        print('train.csv parsing....')
        train_target_set = get_feature_set(train_csv, header_name, action_type)
        print('test.csv parsing....')
        test_target_set = get_feature_set(test_csv, header_name, action_type)
        target_set = train_target_set | test_target_set

    target_list = sorted(list(target_set))
    print(save_path, '\nsize', len(target_list), '\n', target_list[0:10])

    save_pickle(save_path, target_list)

    return target_list

def load_pickle(pickle_path):
    print("loading...", pickle_path)
    cur_time = time.time()
    with open(pickle_path, 'rb') as f :
        data = pickle.load(f)
    print('loading done time(s)', int(time.time() - cur_time))
    return data

def load_multiple_pickle(pickle_path):
    print("loading multiple pickle", pickle_path)
    all_path_str = add_str_to_path(pickle_path, '*')
    all_path = glob.glob(all_path_str)
    all_path.sort()
    data = []
    for path in all_path :
        cur_data = load_pickle(path)
        data.extend(cur_data)
    return data

def save_hdf(path, data_name, data):
    print("saving...", path)
    cur_time = time.time()
    save = h5py.File(path, "a")
    save.create_dataset(data_name, data=data)
    save.close()
    print('saving done time(s)', int(time.time() - cur_time))

def load_hdf(path, data_name) :
    print("loading...", path)
    cur_time = time.time()
    f = h5py.File(path, "r")
    '''
    a_group_key = list(f.keys())
    print('a_group_key', a_group_key)
    '''
    data = f[data_name]
    print('loading done time(s)', int(time.time() - cur_time))
    return data

def save_pickle(pickle_path, data):
    print("saving...", pickle_path)
    cur_time = time.time()
    try : 
        with open(pickle_path, 'wb') as f :
            pickle.dump(data, f)
            print('saving done time(s)', int(time.time() - cur_time))

    except MemoryError :
        print("saving FAIL. Diving data...")
        path1 = add_str_to_path(pickle_path, '_1')
        path2 = add_name(pickle_path, '_2')
        size = int(len(data) /2)
        save_pickle(path1, data[:size])
        save_pickle(path2, data[size:])
        
def make_target_idx_list(lookup_list, targets):
    target_idx_list = []
    for target in targets :
        if(target == '') : 
            break;
        
        #target_idx = lookup_list.index(target)
        target_idx = get_index(lookup_list, target)
        target_idx_list.append(target_idx)

    return target_idx_list

def get_one_hot_encoding(target, lookup_list):
    if(target ==''):
        value = 1.0/len(lookup_list)
        return [[value] * len(lookup_list)]
    lookup_size = len(lookup_list)
    target_tokens = target.split('|')
    target_idx_list = make_target_idx_list(lookup_list, target_tokens)
    one_hot = np.eye(lookup_size)[target_idx_list]
    encoding = one_hot

    #return encoding.tolist()
    return encoding

def get_one_hot_sum(target, lookup_list):
    if(target ==''):
        value = 0
        return [[value] * len(lookup_list)]
    one_hot = get_one_hot_encoding(target, lookup_list)
    #one_hot_sum = np.sum(np.array(one_hot), axis = 0)
    one_hot_sum = np.sum(one_hot, axis = 0)
    
    #return [one_hot_sum.tolist()]
    return one_hot_sum

def get_normalized_one_hot_sum(target, lookup_list):
    if(target ==''):
        value = 1.0/len(lookup_list)
        return [[value] * len(lookup_list)]
    one_hot = get_one_hot_encoding(target, lookup_list)
    one_hot_sum = np.sum(np.array(one_hot), axis = 0)
    normalized_one_hot_sum = one_hot_sum / np.sum(one_hot_sum)
    
    return [normalized_one_hot_sum.tolist()]

def get_index(a, x) :
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

def get_item_encoding(item_id_list, item_property_lookup, item_idx):
    property_idx = get_index(item_id_list, int(item_idx))
    property_encoding = item_property_lookup[property_idx]
    return property_encoding

def get_interaction_action_encoding(reference, action_type, item_idx):
    cnt_list = np.zeros((4,), dtype = int)
    np_reference = np.array(reference)
    np_action_type = np.array(action_type)
    reference_matching_idx =(np_reference == item_idx) 
    if np.sum(reference_matching_idx) == 0:
        return cnt_list.tolist()
    reference_matching_action_type = np.extract(reference_matching_idx, np_action_type)

    for action in reference_matching_action_type :
        if action == 'interaction item rating':
            cnt_list[interaction_type.RATING.value] = cnt_list[interaction_type.RATING.value] + 1
        elif action == 'interaction item info':
            cnt_list[interaction_type.INFO.value] = cnt_list[interaction_type.INFO.value] + 1
        elif action == 'interaction item image':
            cnt_list[interaction_type.IMAGE.value] = cnt_list[interaction_type.IMAGE.value] + 1
        elif action == 'interaction item deals':
            cnt_list[interaction_type.DEALS.value] = cnt_list[interaction_type.DEALS.value] + 1

    '''
    if(np.sum(cnt_list)) :
        cnt_list = cnt_list / np.sum(cnt_list)
    '''
    
    return cnt_list.tolist()

def get_grade_price(prices):
    price_list = prices.split('|')
    price_list = [float(price) for price in price_list]
    sort_price = sorted(price_list, reverse=True)
    grade_list = [ sort_price.index(float(price)) for  price in price_list] 
    return grade_list

def get_norm_price(prices):
    price_list = prices.split('|')
    price_list = [float(price) for price in price_list]
    max_price = max(price_list)
    norm_price = [1 - price/max_price for price in price_list]
    
    return norm_price

def flat_list(list_2d):
    result = []
    for list_1d  in list_2d :
        for ele in list_1d :
            result.append(ele)

    return result

def get_click_out_item_list(action_type, reference):
    click_out_item_id = [ reference[i] for i, action in enumerate(action_type) if (action == 'clickout item')]
    return click_out_item_id

def get_session_encoding(session_id, action_type, reference, impressions , prices, item_id_list, click_not_in_impression, is_test = 0):
    #norm_price = get_grade_price(prices)
    norm_price = [int(price) for price in prices.split('|')]
    display_orders = sorted(list(range(len(norm_price))), reverse=True)
    clicked_out_item = get_click_out_item_list(action_type, reference)
    impression_list = impressions.split('|')

    skip_item_list = []
    session_encoding = []
    for item, price, display_order in zip(impression_list, norm_price, display_orders) :
        try : 
            item_id_idx = get_index(item_id_list, item)
        except ValueError:
            item_id_idx = -1
            skip_item_list.append({'session_id':session_id, 'item':item})
        interaction_encoding = get_interaction_action_encoding(reference, action_type, item)
        y_encoding = 1 if item in clicked_out_item else 0

        impression_encoding = [item_id_idx, price, display_order]
        impression_encoding.extend(interaction_encoding)
        impression_encoding.append(y_encoding)

        session_encoding.append(impression_encoding)

    if(is_test) :
        click_item_idx = -1
    else :
        try : 
            click_item_idx = impression_list.index(clicked_out_item[-1])
        except ValueError :
            click_item_idx = -1
            click_not_in_impression.append({'session_id':session_id, 'item' : clicked_out_item[-1]})
            
    return session_encoding, skip_item_list, click_item_idx
 
'''
def get_session_encoding(session_id, action_type, reference, platform, device, current_filters,impressions , prices, item_id_list, item_lookup, click_not_in_impression, is_test = 0):
    session_encoding = []
    #filter_encoding = get_normalized_one_hot_sum(current_filters[-1], filter_lookup)[0]
    filter_encoding = get_one_hot_sum(current_filters[-1], filter_list)[0]
    device_encoding = get_one_hot_encoding(device[-1], device_list)[0]
    platform_encoding = get_one_hot_encoding(platform[-1], platform_list)[0]
    #norm_price = get_norm_price(prices[-1])
    norm_price = get_grade_price(prices[-1])

    clicked_out_item = get_click_out_item_list(action_type, reference)
    impression_list = impressions[-1].split('|')

    skip_item_list = []
    for item, price in zip(impression_list, norm_price) :
        try : 
            item_encoding = get_item_encoding(item_id_list, item_lookup, item)
        except ValueError:
            item_encoding = [0.0] * feature_size.ITEM_PROPERTIES.value
            skip_item_list.append({'session_id':session_id, 'item':item})
        interaction_encoding = get_interaction_action_encoding(reference, action_type, item)
        price_encoding = [price]
        y_encoding = [1.0] if item in clicked_out_item else [0.]

        cur_encoding = flat_list([filter_encoding, interaction_encoding, device_encoding, platform_encoding, item_encoding, price_encoding, y_encoding])
        session_encoding.append(cur_encoding)
    if(is_test) :
        click_item_idx = -1
    else :
        try : 
            click_item_idx = impression_list.index(clicked_out_item[-1])
        except ValueError :
            click_item_idx = -1
            click_not_in_impression.append({'session_id':session_id, 'item' : clicked_out_item[-1]})
            
    return session_encoding, skip_item_list, click_item_idx
'''
    
'''
def add_data(dic, enc, session_id, cur_enc, click_item_idx, user_id, timestamp, step):
    idx_start = dic[-1]["index"].stop if dic else 0
    idx_end = idx_start + len(cur_enc)
    _slice = slice(idx_start, idx_end)
    session_dict = {"index" : _slice, "click" : click_item_idx, "user_id" : user_id, "session_id" : session_id, "timestamp" : timestamp, "step" :  step}
    dic.append(session_dict)            
    enc[_slice] = cur_enc
'''

def add_data(dic, enc, cur_enc, filter_idx, platform_idx, device_idx, user_id_idx, session_id_idx, timestamp, step, click_item_idx, impressions):
    idx_start = dic[-1]["slice"].stop if dic else 0
    idx_end = idx_start + len(cur_enc)
    _slice = slice(idx_start, idx_end)
    session_dict = {"slice" : _slice, "filter_idx" : filter_idx, "platform_idx":platform_idx, "device_idx":device_idx, "user_id_idx" : user_id_idx, "session_id_idx" : session_id_idx, "timestamp" : timestamp, "step" :  step, "click" : click_item_idx, "impressions" : impressions}
    dic.append(session_dict)            
    enc[_slice] = cur_enc

def make_new_data_in_hdf(path, data_name, shape):
    f = h5py.File(path, "w")
    data = f.create_dataset(data_name, shape, maxshape = shape)
    return data

def make_empty_hdf(path, data_name, shape, dtype = 'f'):
    f = h5py.File(path, "a")
    data = f.create_dataset(data_name, shape, maxshape = shape, dtype = dtype)
    return data

def resize_hdf(data, shape):
    data.resize(shape)

def get_csv_size(path) :
    f = open(path, 'r')
    lines = csv.reader(f)
    header = next(lines, None)
    lines_size = sum(1 for line in lines)
    f.close()

    return lines_size
     
def add_impression_to_dic(csv_path, dic_path) :
    #lines_size = get_csv_size(csv_path)
    lines_size = 3782335
    f = open(csv_path, 'r')
    lines = csv.reader(f)
    header = next(lines, None)
    print('lines_size', lines_size)
    dic = load_pickle(dic_path)
 
    complete_cnt = 0
    session_id, step, action_type, impressions = [], [], [], []
    for i, line in enumerate(lines) : 
        session_id.append(line[header_idx.SESSION_ID.value])
        step.append(line[header_idx.STEP.value])
        action_type.append(line[header_idx.ACTION_TYPE.value])
        impressions.append(line[header_idx.IMPRESSIONS.value])
        
        if(i == 0) : continue
        if (step[-1] =='1') or (i == lines_size-1) :
            last_idx = -1 if step[-1] == '1' else None
            if(action_type[-2] != 'clickout item'):
                del session_id[:last_idx], action_type[:last_idx], step[:last_idx], impressions[:last_idx]
                continue

            cur_session_id = session_id[-2]
            cur_impressions = impressions[-2].split('|')
            if(dic[complete_cnt]["session_id"] != cur_session_id):
                print('cur_dic["session_id"] != cur_session_id')
                return 
            dic[complete_cnt]["item_recommendations"] = cur_impressions
            complete_cnt += 1
            
            del session_id[:last_idx], action_type[:last_idx], step[:last_idx], impressions[:last_idx]
    save_pickle(dic_path, dic) 
 
def parse_csv(csv_path, item_id_list_path, session_id_list_path, user_id_list_path, filter_list_path, platform_list_path, device_list_path, train_val_encdoing_path, train_dic_path, val_dic_path, skip_item_path, skip_session_path, click_not_in_impression_path, is_test = 0):
    lines_size = get_csv_size(csv_path)
    f = open(csv_path, 'r')
    lines = csv.reader(f)
    header = next(lines, None)
    print('lines_size', lines_size)
    
    item_id_list = load_pickle(item_id_list_path)
    session_id_list = load_pickle(session_id_list_path)
    user_id_list = load_pickle(user_id_list_path)
    filter_list = load_pickle(filter_list_path)
    platform_list = load_pickle(platform_list_path)
    device_list = load_pickle(device_list_path)

    user_id, session_id, timestamp, step, action_type, reference, platform, city, device, current_filters, impressions, prices = ([] for i in range(12) )

    skip_item_list, skip_session_list, click_not_in_impression_list = [], [], []
    train_dic, val_dic = [], []
    train_sess_cnt, val_sess_cnt = 0, 0
    if os.path.exists(train_val_encdoing_path) : os.remove(train_val_encdoing_path) 
    train_enc = make_empty_hdf(train_val_encdoing_path, 'train_enc', (2**30, feature_size.WHOLE_SIZE.value), 'i8')
    val_enc = make_empty_hdf(train_val_encdoing_path, 'val_enc', (2**30, feature_size.WHOLE_SIZE.value), 'i8')

    print('encoding session...')
    cnt_dic = 0
    for i, line in enumerate(lines) : 
        user_id.append(line[header_idx.USER_ID.value])
        timestamp.append(line[header_idx.TIMESTAMP.value])
        session_id.append(line[header_idx.SESSION_ID.value])
        step.append(line[header_idx.STEP.value])
        action_type.append(line[header_idx.ACTION_TYPE.value])
        reference.append(line[header_idx.REFERENCE.value])
        platform.append(line[header_idx.PLATFORM.value])
        device.append(line[header_idx.DEVICE.value])
        current_filters.append(line[header_idx.CURRENT_FILTERS.value])
        impressions.append(line[header_idx.IMPRESSIONS.value])
        prices.append(line[header_idx.PRICES.value])
        
        if(i == 0) : continue
        if (step[-1] =='1') or (i == lines_size-1) :
            last_idx = -1 if step[-1] == '1' else None
            if(action_type[-2] != 'clickout item'):
                skip_session_list.append(session_id[0])
                del session_id[:last_idx], step[:last_idx], action_type[:last_idx], reference[:last_idx], platform[:last_idx], device[:last_idx], current_filters[:last_idx], impressions[:last_idx], prices[:last_idx]
                continue
 
            cur_timestamp = timestamp[-2]
            cur_step = step[-2]
            cur_impressions = impressions[-2]
            cur_session_id = session_id[-2]
            cur_prices = prices[-2]
            cur_session_id_idx = get_index(session_id_list, cur_session_id)
            cur_user_id_idx = get_index(user_id_list, user_id[-2])
            cur_filter_idx = [get_index(filter_list, cur_filter) for cur_filter in current_filters[-2].split('|') if cur_filter != '']
            cur_platform_idx = get_index(platform_list, platform[-2])
            cur_device_idx = get_index(device_list, device[-2])
 
            cur_action_type = action_type[:last_idx]
            cur_reference = reference[:last_idx]
            
            cur_session_encoding, skip_items, click_item_idx = get_session_encoding(cur_session_id, cur_action_type, cur_reference ,cur_impressions , cur_prices, item_id_list, click_not_in_impression_list, is_test = is_test)

            if( is_test==0 and  np.random.rand() < 0.01):
                add_data(val_dic, val_enc, cur_session_encoding, cur_filter_idx, cur_platform_idx, cur_device_idx, cur_user_id_idx, cur_session_id_idx, cur_timestamp, cur_step, click_item_idx, cur_impressions) 
            else :
                add_data(train_dic, train_enc, cur_session_encoding, cur_filter_idx, cur_platform_idx, cur_device_idx, cur_user_id_idx, cur_session_id_idx, cur_timestamp, cur_step, click_item_idx, cur_impressions) 

            skip_item_list.extend(skip_items)

            del user_id[:last_idx],timestamp[:last_idx],session_id[:last_idx], step[:last_idx], action_type[:last_idx], reference[:last_idx], platform[:last_idx], device[:last_idx], current_filters[:last_idx], impressions[:last_idx], prices[:last_idx]

            '''
            if(len(train_dic) == 1000):
                break;
            '''

        if(i % 100000 == 0):
            print('line', i + 1,'/', lines_size, int(float(i+1)/lines_size*100) , '% Done')
            #print('train_dic', len(train_dic),'/', 10000, int(len(train_dic)/10000*100) , '% Done')
    f.close()
    del item_id_list

    train_enc_size = train_dic[-1]["slice"].stop
    resize_hdf(train_enc, (train_enc_size, feature_size.WHOLE_SIZE.value))
    val_enc_size = val_dic[-1]["slice"].stop if val_dic else 0
    resize_hdf(val_enc, (val_enc_size, feature_size.WHOLE_SIZE.value))

    print('encoding session Done')
    print('# of train encoding', train_enc_size, '# of train session', len(train_dic))
    print('# of val encoding', val_enc_size, '# of val session', len(val_dic))
    print('# of skip item', len(skip_item_list), '\n', skip_item_list[0:10])
    print('# of skip session', len(skip_session_list), '\n', skip_session_list[0:10])
    print('# of click_not_in_impression', len(click_not_in_impression_list), '\n', click_not_in_impression_list[0:10])

    save_pickle(train_dic_path, train_dic)
    save_pickle(val_dic_path, val_dic)
    save_pickle(skip_item_path,skip_item_list)
    save_pickle(skip_session_path, skip_session_list)
    save_pickle(click_not_in_impression_path, click_not_in_impression_list)

'''
def parse_csv(csv_path, item_id_list_path, item_property_encoding_path, train_val_encdoing_path, train_dic_path, val_dic_path, skip_item_path, skip_session_path, click_not_in_impression_path, is_test = 0):
    lines_size = get_csv_size(csv_path)
    f = open(csv_path, 'r')
    lines = csv.reader(f)
    header = next(lines, None)
    print('lines_size', lines_size)
    
    item_id_list = load_pickle(item_id_list_path)
    item_property_encoding = load_pickle(item_property_encoding_path)
    #item_property_encoding = 0 

    user_id, session_id, timestamp, step, action_type, reference, platform, city, device, current_filters, impressions, prices = ([] for i in range(12) )

    skip_item_list, skip_session_list, click_not_in_impression_list = [], [], []
    train_dic, val_dic = [], []
    train_sess_cnt, val_sess_cnt = 0, 0
    if os.path.exists(train_val_encdoing_path) : os.remove(train_val_encdoing_path) 
    train_enc = make_empty_hdf(train_val_encdoing_path, 'train_enc', (2**30, feature_size.WHOLE_SIZE.value))
    val_enc = make_empty_hdf(train_val_encdoing_path, 'val_enc', (2**30, feature_size.WHOLE_SIZE.value))

    print('encoding session...')
    cnt_dic = 0
    for i, line in enumerate(lines) : 
        user_id.append(line[header_idx.USER_ID.value])
        timestamp.append(line[header_idx.TIMESTAMP.value])
        session_id.append(line[header_idx.SESSION_ID.value])
        step.append(line[header_idx.STEP.value])
        action_type.append(line[header_idx.ACTION_TYPE.value])
        reference.append(line[header_idx.REFERENCE.value])
        platform.append(line[header_idx.PLATFORM.value])
        device.append(line[header_idx.DEVICE.value])
        current_filters.append(line[header_idx.CURRENT_FILTERS.value])
        impressions.append(line[header_idx.IMPRESSIONS.value])
        prices.append(line[header_idx.PRICES.value])
        
        if(i == 0) : continue
        if (step[-1] =='1') or (i == lines_size-1) :
            last_idx = -1 if step[-1] == '1' else None
            if(action_type[-2] != 'clickout item'):
                skip_session_list.append(session_id[0])
                del session_id[:last_idx], step[:last_idx], action_type[:last_idx], reference[:last_idx], platform[:last_idx], device[:last_idx], current_filters[:last_idx], impressions[:last_idx], prices[:last_idx]
                continue
 
            cur_session_id = session_id[-2]
            cur_user_id = user_id[-2]
            cur_timestamp = timestamp[-2]
            cur_step = step[-2]

            cur_action_type = action_type[:last_idx]
            cur_reference = reference[:last_idx]
            cur_platform = platform[:last_idx]
            cur_device = device[:last_idx]
            cur_current_filters = current_filters[:last_idx]
            cur_prices = prices[:last_idx]
            

            cur_session_encoding, skip_items, click_item_idx = get_session_encoding(cur_session_id, cur_action_type, cur_reference, cur_platform, cur_device, cur_current_filters,cur_impressions , cur_prices, item_id_list, item_property_encoding, click_not_in_impression_list, is_test = is_test)

            if(np.random.rand() < 0.05):
                add_data(val_dic, val_enc,cur_session_id, cur_session_encoding, click_item_idx, cur_user_id, cur_timestamp, cur_step) 
            else :
                add_data(train_dic, train_enc, cur_session_id, cur_session_encoding, click_item_idx, cur_user_id, cur_timestamp, cur_step) 

            skip_item_list.extend(skip_items)

            del user_id[:last_idx],timestamp[:last_idx],session_id[:last_idx], step[:last_idx], action_type[:last_idx], reference[:last_idx], platform[:last_idx], device[:last_idx], current_filters[:last_idx], impressions[:last_idx], prices[:last_idx]

            if(len(train_dic) == 10000):
                break;

        if(i % 10000 == 0):
            print('line', i + 1,'/', lines_size, int(float(i+1)/lines_size*100) , '% Done')
            print('train_dic', len(train_dic),'/', 10000, int(len(train_dic)/10000*100) , '% Done')
    f.close()
    del item_id_list, item_property_encoding

    train_enc_size = train_dic[-1]["index"].stop
    resize_hdf(train_enc, (train_enc_size, feature_size.WHOLE_SIZE.value))
    val_enc_size = val_dic[-1]["index"].stop if val_dic else 0
    resize_hdf(val_enc, (val_enc_size, feature_size.WHOLE_SIZE.value))

    print('encoding session Done')
    print('# of train encoding', train_enc_size, '# of train session', len(train_dic))
    print('# of val encoding', val_enc_size, '# of val session', len(val_dic))
    print('# of skip item', len(skip_item_list), '\n', skip_item_list[0:10])
    print('# of skip session', len(skip_session_list), '\n', skip_session_list[0:10])
    print('# of click_not_in_impression', len(click_not_in_impression_list), '\n', click_not_in_impression_list[0:10])

    save_pickle(train_dic_path, train_dic)
    save_pickle(val_dic_path, val_dic)
    save_pickle(skip_item_path,skip_item_list)
    save_pickle(skip_session_path, skip_session_list)
    save_pickle(click_not_in_impression_path, click_not_in_impression_list)
'''

def parse_meta_item(csv_file, property_list_path, item_id_list_path, item_proprety_encoding_path):
    property_list = load_pickle(property_list_path)
    item_id_list = load_pickle(item_id_list_path)
    line_size = get_csv_size(csv_file)
    meta_item_list = []
    val_enc = make_empty_hdf(item_proprety_encoding_path, 'item_property_binary', (line_size, feature_size.ITEM_PROPERTIES.value))

    f = open(csv_file, 'r')
    lines = csv.reader(f)
    header = next(lines, None)

    for i, line in enumerate(lines) : 
        item_id = line[header_idx.ITEM_ID.value]
        item_properties = line[header_idx.ITEM_PROPERTIES.value]
        #cur_property_encoding = get_normalized_one_hot_sum(item_properties, property_lookup)
        item_id_idx = get_index(item_id_list, item_id)
        item_property_binary = get_one_hot_sum(item_properties, property_list)
    
        '''
        cur_meta_item_list = cur_property_encoding[0]
        cur_meta_item_list.insert(0, int(item_id))

        meta_item_list.append(cur_meta_item_list)
        '''
        val_enc[item_id_idx] = item_property_binary

        if(i % 10000 == 0) : 
            print(i + 1,'/', data_size.ITEM.value, int(float(i+1)/line_size*100) , '% Done')

        '''
        if(i == 15000):
            break
        '''
    f.close()

    '''
    meta_item_list.sort()
    np_meta_item = np.array(meta_item_list)
    del meta_item_list
    item_id_list = np_meta_item[:, 0].tolist()
    item_properties_encoding = np_meta_item[:, 1:].tolist()

    save_pickle(item_id_list_path, item_id_list)
    save_pickle(item_proprety_encoding_path, item_properties_encoding)
    '''

def decode_batch(filter_idx, platform_idx, device_idx, price, dp_order, interaction, y, user_id_idx, item_id_idx, click, platform_list_path, device_list_path, filter_list_path, user_id_list_path, session_id_list_path, item_id_list_path) :
    platform_list = load_pickle(platform_list_path)
    device_list = load_pickle(device_list_path)
    filter_list = load_pickle(filter_list_path)
    user_id_list = load_pickle(user_id_list_path)
    session_id_list = load_pickle(session_id_list_path)
    item_id_list = load_pickle(item_id_list_path)

    filters = [filter_list[i] for i in filter_idx]
    platform = platform_list[platform_idx]
    device = device_list[device_idx]
    user_id = user_id_list[user_id_idx]
    click = item_id_list[item_id_idx[click]]

    session_info = {'user_id' :user_id, 'platform' : platform, 'device' : device, 'filters' : filters, 'click' : click}
    impression_list = []
    for _item_id_idx, _price, _dp_order, _interaction, _y in zip(item_id_idx, price, dp_order, interaction, y) :
        item_id = item_id_list[_item_id_idx]
        imp_dic = {'item_id' : item_id, 'interaction' : _interaction, 'price' : _price , 'dp_order' :_dp_order, 'y':_y}
        impression_list.append(imp_dic)

    return session_info, impression_list

def decode_dic(dic, enc_path, data_name, platform_list_path, device_list_path, filter_list_path, user_id_list_path, session_id_list_path, item_id_list_path) :
    platform_list = load_pickle(platform_list_path)
    device_list = load_pickle(device_list_path)
    filter_list = load_pickle(filter_list_path)
    user_id_list = load_pickle(user_id_list_path)
    session_id_list = load_pickle(session_id_list_path)
    item_id_list = load_pickle(item_id_list_path)

    filters = [filter_list[i] for i in dic['filter_idx']]
    platform = platform_list[dic['platform_idx']]
    device = device_list[dic['device_idx']]
    user_id = user_id_list[dic['user_id_idx']]
    session_id = session_id_list[dic['session_id_idx']]
    click = dic['impressions'].split('|')[dic['click']]

    session_info = {'user_id' :user_id, 'session_id' : session_id, 'timestamp' : dic['timestamp'], 'step' :  dic['step'], 'platform' : platform, 'device' : device, 'filters' : filters, 'click' : click}
    enc = load_hdf(enc_path, data_name)
    impression_encs = enc[dic['slice']]
    impression_list = []
    for impression_enc in impression_encs :
        item_id = item_id_list[impression_enc[encoding_idx.ITEM_ID.value]]
        price = impression_enc[encoding_idx.PRICE.value]
        dp_order = impression_enc[encoding_idx.DP_ORDER.value]
        interaction = impression_enc[encoding_idx.INTERACTION.value]
        y = impression_enc[encoding_idx.Y.value]

        imp_dic = {'item_id' : item_id, 'interaction' : interaction, 'price' : price , 'dp_order' :dp_order, 'y':y}
        impression_list.append(imp_dic)

    return session_info, impression_list

if __name__ == "__main__" :
    np.random.seed(0)
    item_metadata_csv_path = '/home/sapark/class/ml/final/data/item_metadata.csv'
    train_csv_path = '/home/sapark/class/ml/final/data/train.csv'
    test_csv_path = '/home/sapark/class/ml/final/data/test.csv'

    property_list_path ='/hdd/sap/ml/final/data/pkl/feature_list/property_list.pkl' 
    platform_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/platform_list.pkl'
    device_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/device_list.pkl'
    filter_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/filter_list.pkl'
    poi_list_path ='/hdd/sap/ml/final/data/pkl/feature_list/poi_list.pkl' 
    destination_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/destination_list.pkl'
    user_id_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/user_id_list.pkl'
    city_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/city_list.pkl'

    item_id_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/item_id_list.pkl'
    item_property_encoding_path = '/hdd/sap/ml/final/data/pkl/feature_list/item_proprerty_encoding.hdf5'

    train_val_encoding_path = '/home/sapark/class/ml/final/data/pkl/train_val_encoding.hdf5'
    train_dic_path = '/hdd/sap/ml/final/data/pkl/train_dic.pkl'
    val_dic_path = '/hdd/sap/ml/final/data/pkl/val_dic.pkl'
    skip_item_path = '/hdd/sap/ml/final/data/pkl/skip_item.pkl'
    skip_session_path = '/hdd/sap/ml/final/data/pkl/skip_session.pkl'
    click_not_in_impression_path = '/hdd/sap/ml/final/data/pkl/click_not_in_impression.pkl'

    test_encoding_path = '/home/sapark/class/ml/final/data/pkl/test_encoding.hdf5'
    test_dic_path = '/hdd/sap/ml/final/data/pkl/test_dic.pkl'
    dummy_path = '/hdd/sap/ml/final/data/pkl/dummy.pkl'

    #1. extract feature list
    device_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/device_list.pkl'
    filter_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/filter_list.pkl'
    poi_list_path ='/hdd/sap/ml/final/data/pkl/feature_list/poi_list.pkl' 
    destination_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/destination_list.pkl'
    user_id_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/user_id_list.pkl'
    session_id_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/session_id_list.pkl'
    city_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/city_list.pkl'
    item_id_list_path = '/hdd/sap/ml/final/data/pkl/feature_list/item_id_list.pkl'

    '''
    header_name_list = ['device', 'current_filters', 'user_id', 'session_id', 'item_id']
    action_type_list = [None, 'None', None, None, None]
    is_item_metadata_list = [False, False, False, False, True]
    save_path_list = [device_list_path, filter_list_path, user_id_list_path, session_id_list_path, item_id_list_path]
    for header_name, action_type, is_item_metadata, save_path in zip(header_name_list, action_type_list, is_item_metadata_list, save_path_list) :
        make_feature_list(train_csv_path, test_csv_path, item_metadata_csv_path, header_name, save_path, is_item_metadata, action_type)
    '''

    #2. parse_meta_item csv
    #parse_meta_item(item_metadata_csv_path, property_list_path, item_id_list_path, item_property_encoding_path)
    #item_id = load_pickle(item_id_list_path)
    #item_property = load_pickle(item_property_encoding_path)

    #3. parse_train_csv
    parse_csv(train_csv_path, item_id_list_path, session_id_list_path, user_id_list_path, filter_list_path, platform_list_path, device_list_path, train_val_encoding_path, train_dic_path, val_dic_path, skip_item_path, skip_session_path, click_not_in_impression_path, is_test = 0)
    '''
    train_dic = load_pickle(train_dic_path)
    session_dic, impression_dics = decode_dic(train_dic[100], train_val_encoding_path, 'train_enc', device_list_path, filter_list_path, user_id_list_path, session_id_list_path, item_id_list_path)
    print(session_dic)
    for impression_dic in impression_dics:
        print(impression_dic) 
    '''
    #4. parse_test_csv
    parse_csv(test_csv_path, item_id_list_path, session_id_list_path, user_id_list_path, filter_list_path, platform_list_path, device_list_path, test_encoding_path, test_dic_path, dummy_path, dummy_path, dummy_path, dummy_path, is_test = 1)
    #parse_csv(test_csv_path, item_id_list_path, item_property_encoding_path, test_encoding_path, test_dic_path, dummy_path, dummy_path, dummy_path, dummy_path, is_test = 1)
    #add_impression_to_dic(test_csv_path, test_dic_path)
    #dic = load_pickle(test_dic_path)
    print('aaa')
