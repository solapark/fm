from time import sleep
import os 
import functools
import tensorflow as tf
from common import *
from tensorflow.python import debug as tf_debug

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class CONTEXT_FM:
    def __init__(self, w_dim, lr, loss_type, item_size):
        self.W_dim = w_dim
        self.lr = lr
        self.loss_type = loss_type
        self.item_size = item_size

        with tf.name_scope('input') as scope : 
            self.filter_idx = tf.placeholder(tf.int32, [None, ], name='filter_idx')
            self.platform_idx = tf.placeholder(tf.int32, (), name='platform_idx' )
            self.device_idx = tf.placeholder(tf.int32, () , name='device_idx' )
            self.interaction = tf.placeholder(tf.float32, [None, 4], name = 'interaction')
            self.price = tf.placeholder(tf.float32, [None, ], name = 'price')
            self.order = tf.placeholder(tf.int32, [None, ], name = 'order')
            self.item_property_binary = tf.placeholder(tf.float32, [None, feature_size.ITEM_PROPERTIES.value], name = 'item_property_idx')

            self.W_user_id_ph = tf.placeholder(tf.float32, [self.W_dim, ], name = 'W_user_id')
            self.B_user_id_ph = tf.placeholder(tf.float32, (), name = 'B_user_id')
            self.W_item_id_ph = tf.placeholder(tf.float32, [25, self.W_dim], name = 'W_item_id')
            self.B_item_id_ph = tf.placeholder(tf.float32, [25, ], name = 'B_item_id')

            self.y =tf.placeholder(tf.float32, [None, ], name = 'y')
            self.impression = tf.placeholder(tf.string, [None, ], name = 'impression')
            self.click_idx = tf.placeholder(tf.int32, shape=(), name = 'click_idx') # scalar

        with tf.name_scope('W') as scope : 
            self.W_filter = tf.Variable(tf.truncated_normal([feature_size.FILTER.value, self.W_dim]), name = 'filter')
            self.W_platform = tf.Variable(tf.truncated_normal([feature_size.PLATFORM.value, self.W_dim]), name = 'platform')
            self.W_device = tf.Variable(tf.truncated_normal([feature_size.DEVICE.value, self.W_dim]), name = 'device')
            self.W_interaction = tf.Variable(tf.truncated_normal([feature_size.INTERACTION.value, self.W_dim]), name = 'interaction')
            self.W_price = tf.Variable(tf.truncated_normal([self.W_dim, ]), name = 'price')
            #self.W_order = tf.Variable(tf.truncated_normal([self.W_dim, ]), name = 'order')
            self.W_order = tf.Variable(tf.truncated_normal([25, self.W_dim, ]), name = 'order')
            self.W_item_property = tf.Variable(tf.truncated_normal([feature_size.ITEM_PROPERTIES.value, self.W_dim]), name = 'item_property')
            self.W_user_id = tf.Variable(tf.constant(0.0, shape = [self.W_dim, ] ), name = 'user_id')
            self.W_item_id = tf.Variable(tf.constant(0.0, shape = [self.item_size, self.W_dim] ), name = 'itme_id') 

        with tf.name_scope('B') as scope : 
            self.B_filter = tf.Variable(tf.truncated_normal([feature_size.FILTER.value, ]), name = 'filter')
            self.B_platform = tf.Variable(tf.truncated_normal([feature_size.PLATFORM.value, ]), name = 'platform')
            self.B_device = tf.Variable(tf.truncated_normal([feature_size.DEVICE.value, ]), name = 'device')
            self.B_interaction = tf.Variable(tf.truncated_normal([feature_size.INTERACTION.value, ]), name = 'interaction')
            self.B_price = tf.Variable(tf.truncated_normal(shape = ()), name = 'price')
            #self.B_order = tf.Variable(tf.truncated_normal(shape = ()), name = 'order')
            self.B_order = tf.Variable(tf.truncated_normal(shape = [25, ]), name = 'order')
            self.B_item_property = tf.Variable(tf.truncated_normal([feature_size.ITEM_PROPERTIES.value, ]), name = 'item_property')
            self.B_user_id = tf.Variable(tf.constant(0.0), name = 'user_id')
            self.B_item_id = tf.Variable(tf.constant(0.0, shape = [self.item_size, ]), name = 'item_id') 
            self.B_global = tf.Variable(tf.truncated_normal(shape = ()), name = 'global') 

        self.define_emb_W()
        self.define_emb_B()
        self.logits
        self.loss
        self.temp_loss
        self.optimize
        self.optimize_user
        self.rank
        self.prediction
        self.reciprocal_rank
        self.assign_var
        
    @lazy_property
    def assign_var(self) :
        ass_W_user = tf.assign(self.W_user_id, self.W_user_id_ph)
        ass_B_user = tf.assign(self.B_user_id, self.B_user_id_ph)
        #ass_W_item = tf.assign(self.W_item_id, self.W_item_id_ph)
        #ass_B_item = tf.assign(self.B_item_id, self.B_item_id_ph)
        #return ass_W_user, ass_B_user, ass_W_item, ass_B_item
        return ass_W_user, ass_B_user
        
    def define_emb_W(self) :
        num_item = tf.shape(self.interaction)[0]
        with tf.name_scope('emb_W') as scope : 
            self.filter_emb_W = tf.reduce_sum(tf.nn.embedding_lookup(self.W_filter, self.filter_idx), 0) #(w_dim, )
            self.platform_emb_W = tf.nn.embedding_lookup(self.W_platform, self.platform_idx) #(w_dim, )
            self.device_emb_W = tf.nn.embedding_lookup(self.W_device, self.device_idx) #(w_dim, )
            self.interaction_emb_W = tf.matmul(self.interaction, self.W_interaction) #(None, 4) * (4, w_dim) = (None, w_dim)
            #price_emb_W = tf.expand_dims(self.price, 1) * tf.expand_dims(self.W_price, 0) #(None, w_dim)
            price_exp_W = tf.expand_dims(tf.expand_dims(self.price, 0), 2) #(1, None, 1)
            price_conv = tf.layers.conv1d(price_exp_W, self.W_dim, 5, padding = 'same', activation=tf.nn.relu) # (1, none, w_dim)
            price_fc = tf.contrib.layers.fully_connected(price_conv, self.W_dim) # (1, none, w_dim) 
            self.price_emb_W = tf.reshape(price_fc, [-1, self.W_dim])[:num_item] # (none, w_dim) 
            self.order_emb_W = tf.nn.embedding_lookup(self.W_order, self.order) #(None, w_dim)
            self.item_property_emb_W = tf.matmul(self.item_property_binary, self.W_item_property) #(None, w_dim)

    def define_emb_B(self) :
        with tf.name_scope('emb_B') as scope : 
            self.filter_emb_B = tf.reduce_sum(tf.nn.embedding_lookup(self.B_filter, self.filter_idx)) # [1,]
            self.platform_emb_B = tf.nn.embedding_lookup(self.B_platform, self.platform_idx) #[1,]
            self.device_emb_B = tf.nn.embedding_lookup(self.B_device, self.device_idx) #(1, )
            self.interaction_emb_B = tf.reduce_sum( tf.matmul(self.interaction, tf.expand_dims(self.B_interaction, 1)), 1) #(None, 4) * (4, 1) = (None, 1) -> (None, )
            self.price_emb_B = self.price * self.B_price #(None, )
            self.order_emb_B = tf.nn.embedding_lookup(self.B_order, self.order) #(None, )
            self.item_property_emb_B = tf.reduce_sum(tf.matmul(self.item_property_binary, tf.expand_dims(self.B_item_property, 1)), 1) #(None, 157) * (157, 1) = (None, 1) -> (None, )

    def choose_emb_W(self):
        num_item = tf.shape(self.interaction)[0]
        return [self.platform_emb_W, self.device_emb_W, self.interaction_emb_W, self.price_emb_W, self.item_property_emb_W, self.W_item_id[:num_item]] 

    def choose_emb_B(self):
        num_item = tf.shape(self.interaction)[0]
        return [self.B_global, self.platform_emb_B,  self.device_emb_B,  self.interaction_emb_B, self.price_emb_B,  self.item_property_emb_B, self.B_item_id[:num_item]]

    def calc_fm(self, emb_W_list, emb_B_list) :
        with tf.name_scope('first_term') as scope : 
            emb_W_sum = tf.constant(0.0, shape = [self.W_dim, ])  
            for emb_W in emb_W_list : emb_W_sum += emb_W
            emb_W_sum_square_sum = tf.reduce_sum( tf.square(emb_W_sum) , axis = 1)  #(None, )

        with tf.name_scope('second_term') as scope : 
            emb_W_sum_square = tf.constant(0.0, shape = ())  
            for emb_W in emb_W_list : emb_W_sum_square += tf.reduce_sum(tf.square(emb_W), axis = -1)

        with tf.name_scope('emb_B_sum') as scope : 
            emb_B_sum = tf.constant(0.0, shape = [self.W_dim, ])  
            for emb_B in emb_B_list : emb_B_sum += emb_B

        with tf.name_scope('logit') as scope : 
            #logit = emb_W_sum_square_sum - emb_W_sum_square + emb_B
            logit = emb_W_sum_square_sum - emb_W_sum_square

        return logit

    '''
    @lazy_property
    def logits(self):
        emb_W_list = self.choose_emb_W()
        emb_B_list = self.choose_emb_B()
        logits = self.calc_fm(emb_W_list, emb_B_list)
        return logits
    '''

    @lazy_property
    #def logits_with_user(self):
    def logits(self):
        emb_W_list = self.choose_emb_W()
        emb_B_list = self.choose_emb_B()
        new_emb_W_list = []
        new_emb_B_list = []
        for emb_W in emb_W_list :
            new_emb_W_list.append(emb_W * self.W_user_id)
        for emb_B in emb_B_list :
            new_emb_B_list.append(emb_B * self.B_user_id)
              
        logits = self.calc_fm(new_emb_W_list, new_emb_B_list)
        return logits
        
    def loss_BPR(self, T_logit, F_logit):
        T_exp = tf.expand_dims(T_logit, 1) # (# of T, 1)
        F_exp = tf.expand_dims(F_logit, 0) # (1, # of F)

        TFsub = tf.subtract(T_exp, F_exp) # (# of T, # of F)
        sig_TFsub = tf.sigmoid(TFsub)
        log = tf.log(sig_TFsub + 1.0e-10) # (# of T, # of F)

        loss  = -tf.reduce_mean(log) # scalar

        return loss
        #return T_exp, F_exp, TFsub, sig_TFsub, log, loss

    def loss_TOP1(self, T_logit, F_logit):
        F_exp = tf.expand_dims(F_logit, 1) # (# of F, 1)
        T_exp = tf.expand_dims(T_logit, 0) # (1, # of T)

        TFsub = tf.subtract(F_exp, T_exp) # (# of F, # of T)
        TFsub_reg = tf.sigmoid(TFsub) + tf.sigmoid(tf.square(F_exp)) # (# of F, # of T)

        loss  = tf.reduce_mean(TFsub_reg) # scalar
        #return loss
        return T_exp, F_exp, TFsub, TFsub_reg, loss

    @lazy_property
    def temp_loss(self):
        '''
        logit_min = tf.reduce_min(tf.abs(self.logits))
        logit_max = tf.reduce_max(tf.abs(self.logits))
        logit_max_sub_min = tf.maximum(logit_max-logit_min, 1)
        #norm_logit = (self.logits - logit_min )/ (logit_max - logit_min) 
        '''
        norm_logit = self.logits / tf.reduce_max(tf.abs(self.logits)) * 2 
        T_idx = tf.equal(self.y, 1.0) # (# of T, )
        F_idx = tf.not_equal(self.y, 1.0) # (# of F)
        #T_logit = tf.boolean_mask(self.logits, T_idx)# (# of T, )
        T_logit = tf.boolean_mask(norm_logit, T_idx)# (# of T, )
        T_logit = tf.cond(tf.equal(tf.size(T_logit),0), lambda: tf.constant([0.0]), lambda: T_logit)
        #F_logit = tf.boolean_mask(self.logits, F_idx)# (# of F)
        F_logit = tf.boolean_mask(norm_logit, F_idx)# (# of F)
        F_logit = tf.cond(tf.equal(tf.size(F_logit),0), lambda: tf.constant([0.0]), lambda: F_logit)

        if(self.loss_type == 'BPR') :
            loss = self.loss_BPR(T_logit, F_logit)  
            #T_exp, F_exp, TFsub, sig_TFsub, log, loss = self.loss_BPR(T_logit, F_logit)  
        elif(self.loss_type == 'TOP1') :
            #loss = self.loss_TOP1(T_logit, F_logit) 
            T_exp, F_exp, TFsub, TFsub_reg, loss = self.loss_TOP1(T_logit, F_logit)  
        elif(self.loss_type == 'MS') :
            loss = self.loss_MS() 

        #return T_exp, F_exp, TFsub, sig_TFsub, log, loss
        return T_exp, F_exp, TFsub, TFsub_reg, loss

    @lazy_property
    def loss(self):
        #_, _, _, _, _, loss = self.temp_loss
        _, _, _, _, loss = self.temp_loss
        return loss

    @lazy_property
    def optimize_user(self):
        var_list = [self.W_user_id, self.B_user_id]
        return tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss, var_list = var_list)

    @lazy_property
    def optimize(self):
        var_list = tf.trainable_variables()
        del var_list[var_list.index(self.W_user_id)]
        del var_list[var_list.index(self.B_user_id)]
        return tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.loss, var_list = var_list)
        #return  self.optimizer.minimize(self.loss) 
    '''
#    @lazy_property
#    def optimize_ph(self):
        var_list = [self.W_user_id_ph, self.B_user_id_ph, self.W_item_id_ph, self.B_item_id_ph]
        var_list.extend(tf.trainable_variables())
        grad_var_list = self.optimizer.compute_gradients(self.loss, var_list)
        #user_id_item_id = [grad_var_list[i][0] +  grad_var_list[i][1] for i in range(4)]
        #user_id_item_id = [grad_var_list[i][0] for i in range(4)]
        #return self.optimizer.apply_gradients(grad_var_list), user_id_item_id
        #return self.optimizer.apply_gradients(grad_var_list), grad_var_list 
        #return grad_var_list 
        return self.optimizer.apply_gradients(grad_var_list) 
    
    '''
    @lazy_property
    def rank(self):
        return tf.contrib.framework.argsort(self.logits, direction = 'DESCENDING')
        #return tf.argsort(self.logits, direction = 'DESCENDING')

    @lazy_property
    def prediction(self):
        return tf.gather(self.impression, self.rank)
    
    @lazy_property
    def reciprocal_rank(self):
        idx = tf.where(tf.equal(self.rank, self.click_idx)) # (1, 1)
        T_rank = tf.cast(idx[0,0]+1, tf.float32)
        rr = 1.0 / T_rank

        return rr

    def get_W_B_user_id_item_id(self) :
        '''
        dummy = tf.identity(self.W_user_id_ph)
        dummy2 = self.B_user_id_ph + 0
        return tf.gradients(loss, dummy)
        '''
        return self.W_user_id, self.B_user_id, self.W_item_id, self.B_item_id
        #return self.W_user_id_ph, self.B_user_id_ph
        #return self.W_user_id_ph.eval(session =sess), self.B_user_id_ph.eval(session =sess)

    def get_vars(self) :
        #return [{'name':n.name, 'value':n. for n in tf.get_default_graph().as_graph_def().node]
        return tf.trainable_variables()

if __name__ == "__main__" :
    os.environ["CUDA_VISIBLE_DEVICES"]='1' 
    w_dim =10
    lr =10 
    loss_type = 'TOP1'
    item_size = 5

    fm = CONTEXT_FM(w_dim, lr, loss_type, item_size)

    filter_idx = [1,2,3]
    platform_idx = 2
    device_idx = 1
    W_user_id = [0.1]*w_dim
    B_user_id = 10

    price = [1,2,3]
    order = [1,2,3]
    interaction = [[1,0,2,2], [2,2,0,1], [3,4,0,1]]
    W_item_id = [[1]*w_dim, [2]*w_dim, [3]*w_dim, [0]*w_dim, [0]*w_dim]
    B_item_id = [1,2,3, 0, 0]
    item_property_binary = [[1]*4, [0]*4, [1]*4]
    
    y = [1, 0, 0]    
 
    feed_train = {fm.filter_idx : filter_idx, 
                    fm.platform_idx : platform_idx,
                    fm.device_idx : device_idx,
                    fm.interaction : interaction,
                    fm.price : price,
                    fm.order : order,
                    fm.W_user_id_ph : W_user_id,
                    fm.W_item_id_ph : W_item_id,
                    fm.B_user_id_ph : B_user_id,
                    fm.B_item_id_ph : B_item_id,
                    fm.item_property_binary : item_property_binary,
                    fm.y : y
                }

    sess = tf.Session()
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "shark:7000")
    sess.run(tf.global_variables_initializer())
    '''
    for var, val in zip(fm.get_vars(), sess.run(fm.get_vars())) :
        print(var.name)
        print( val)
    '''
    
    #print('user_id', sess.run(fm.get_W_B_user_id()))
    #print('item_id', sess.run(fm.get_W_B_item_id()))
    #sess.run(fm.assign_var, feed_dict = feed_train)
    print('a')
    for i in range(10) :
        sess.run(fm.assign_var, feed_dict = feed_train)
        logit = sess.run(fm.logits, feed_dict = feed_train)
        print('logit', logit)
        #loss = sess.run(fm.loss, feed_dict = feed_train)
        #print('loss', loss)
        T_exp, F_exp, TFsub, TFsub_reg, loss =  sess.run(fm.temp_loss, feed_dict = feed_train)
        '''
        print('T_exp', T_exp)
        print('F_exp', F_exp)
        print('TF_sub', TFsub)
        print('TF_sub_reg', TFsub_reg)
        '''
        print('loss', loss)
        sess.run(fm.optimize, feed_dict = feed_train)
        W_user_id, B_user_id, W_item_id, B_item_id = sess.run(fm.get_W_B_user_id_item_id())
        feed_train[fm.W_user_id_ph] =W_user_id 
        feed_train[fm.W_item_id_ph] =W_item_id 
        feed_train[fm.B_user_id_ph] =B_user_id 
        feed_train[fm.B_item_id_ph] =B_item_id 
 
        #print('W_user_id', W_user_id)
        #print(W_user_id, B_user_id, W_item_id, B_item_id)
 
