
import os
import  numpy as np
import tensorflow as tf
from tacotron.feeder import convert_dur2alignment
os.environ['CUDA_VISIBLE_DEVICES']='0'





a=tf.constant([[1,2,3],[5,6,9]])
b=tf.one_hot(a,7)
print(tf.Session().run(b))