import functools
import tensorflow as tf

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class FM:
    def __init__(self, sess_dim, item_dim, w_dim, lr, gpu_id, loss_type):
        self.session = tf.placeholder(tf.float32, [None, sess_dim])
        self.item = tf.placeholder(tf.float32, [None, item_dim])
        self.y =tf.placeholder(tf.float32, [None, ])
        self.impression = tf.placeholder(tf.string, [None, ])

        self.click_idx = tf.placeholder(tf.int32, shape=()) # scalar

        self.W_session = tf.Variable(tf.truncated_normal([sess_dim, w_dim]))
        self.W_item = tf.Variable(tf.truncated_normal([item_dim, w_dim]))
        self.B_session = tf.Variable(tf.constant(0.1, shape = [sess_dim]))
        self.B_item = tf.Variable(tf.constant(0.1, shape = [item_dim]))
        self.B_global =tf.Variable(tf.constant(0.1))  

        self.lr = lr
        self.loss_type = loss_type
        self.gpu_id = gpu_id

        self.logits
        self.loss
        #self.temp_loss
        self.optimize
        self.rank
        self.prediction
        self.reciprocal_rank
        
    @lazy_property
    def logits(self):
        sess_latent = tf.matmul(self.session, self.W_session, a_is_sparse = True) # (None, w_dim)
        item_latent = tf.matmul(self.item, self.W_item, a_is_sparse = True) # (None, w_dim)
        mul_sess_item = sess_latent * item_latent
        interaction = tf.reduce_sum(mul_sess_item  , axis = 1) # (None, )

        b_sess = tf.reduce_sum(self.session * self.B_session, axis = 1) # (None, )
        b_item = tf.reduce_sum(self.item * self.B_item, axis = 1) # (None, )
        
        y = interaction + b_sess + b_item + self.B_global#(None, )
        return y        

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
        return loss

    @lazy_property
    def loss(self):
    #def temp_loss(self):
        norm_logit = self.logits / tf.reduce_max(tf.abs(self.logits))
        T_idx = tf.equal(self.y, 1.0) # (# of T, )
        F_idx = tf.not_equal(self.y, 1.0) # (# of F)
        T_logit = tf.boolean_mask(norm_logit, T_idx)# (# of T, )
        T_logit = tf.cond(tf.equal(tf.size(T_logit),0), lambda: tf.constant([0.0]), lambda: T_logit)
        F_logit = tf.boolean_mask(norm_logit, F_idx)# (# of F)
        F_logit = tf.cond(tf.equal(tf.size(F_logit),0), lambda: tf.constant([0.0]), lambda: F_logit)

        if(self.loss_type == 'BPR') :
            loss = self.loss_BPR(T_logit, F_logit)  
            #T_exp, F_exp, TFsub, sig_TFsub, log, loss = self.loss_BPR(T_logit, F_logit)  
        elif(self.loss_type == 'TOP1') :
            loss = self.loss_TOP1(T_logit, F_logit) 
        elif(self.loss_type == 'MS') :
            loss = self.loss_MS() 

        return loss
        #return T_exp, F_exp, TFsub, sig_TFsub, log, loss

    '''
    @lazy_property
    def loss(self):
        _, _, _, _, _, loss = self.temp_loss
        return loss
    '''

    @lazy_property
    def optimize(self):
        return  tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.loss) 
    
    @lazy_property
    def rank(self):
        #return tf.argsort(self.logits, direction = 'DESCENDING')
        return tf.contrib.framework.argsort(self.logits, direction = 'DESCENDING')
        #return tf.top_k(self.logits, k = tf.shape(self.logits)[0]).indices

    @lazy_property
    def prediction(self):
        return tf.gather(self.impression, self.rank)
    
    @lazy_property
    def reciprocal_rank(self):
        idx = tf.where(tf.equal(self.rank, self.click_idx)) # (1, 1)
        T_rank = tf.cast(idx[0,0]+1, tf.float32)
        rr = 1.0 / T_rank

        return rr

if __name__ == "__main__" :
    sess_dim = 265
    item_dim = 157
    w_dim = 10
    lr = 0.001
    sigma = 0.1
    loss_type = 'BPR'

    tf.reset_default_graph()
    fm = FM(sess_dim, item_dim, w_dim, lr, sigma, loss_type)
