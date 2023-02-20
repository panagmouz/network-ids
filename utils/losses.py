from keras import backend as K
import tensorflow as tf


def kl_loss(g):
    def kl_loss_num(y_true, y_pred):
        m, s = tf.split(y_pred, num_or_size_splits=2, axis=1, num=None, name='split')
        kl_batch = - .5 * K.sum(1 + s - K.square(m) - K.exp(s), axis=-1)
        return K.mean(kl_batch, axis=-1) * g

    return kl_loss_num
