import tensorflow as tf
from tacotron.models.modules import conv1d
from hparams import hparams


class duration:

    def __init__(self, hparams, training=True):
        self.trainging = training
        self.conv_layer_num = hparams.dur_conv_layer_num
        self.project_dim = hparams.dur_project_dim
        self.conv_kernel_size = hparams.dur_conv_kernel_size
        self.conv_channel = hparams.dur_conv_channel
        self.activation='relu'
        self.dropout_rate=0.5

    def __call__(self,encoder_outputs):
        with tf.variable_scope('duration') as name_scope:
            x=encoder_outputs
            for i in range(self.conv_layer_num):
                x=conv1d(x,kernel_size=self.conv_kernel_size,
                         channels=self.conv_channel,
                         activation=self.activation,
                         is_training=self.trainging,
                         drop_rate=self.dropout_rate,
                         scope='duration_conv%s'%i,
                         bnorm='after')
            x=tf.layers.dense(x,self.project_dim)

        return x


