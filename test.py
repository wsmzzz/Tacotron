import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
if __name__ == '__main__':
    sess=tf.Session()
    a=tf.random.uniform((32,48))
    b=tf.argmax(a,axis=-1)
    print(sess.run(b).shape)