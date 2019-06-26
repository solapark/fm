import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
 
shape=(int(10000),int(10000))
 
with tf.device("/gpu:3"):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)
 
startTime = time.time()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)
 
print("\n" * 2)
print("Time taken:", time.time() - startTime)
print("\n" * 2)
