import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
if __name__ == '__main__':
    align=tf.random.uniform((32,128,16),dtype=tf.float32)
    align=tf.transpose(align,perm=[1,0,2])
    new_T=tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(align)
    one_time=new_T.read(1)
    sess=tf.Session()
    print(sess.run(one_time).shape)
