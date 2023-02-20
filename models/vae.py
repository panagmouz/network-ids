from keras.models import Model, Layer
import keras.backend as K
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input, Multiply, Concatenate
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras import callbacks
from keras.callbacks import CSVLogger
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils import losses
from utils.callbacks import f1_score


class SamplingLayer(Layer):
    def __init__(self):
        self.is_placeholder = True
        super(SamplingLayer, self).__init__()

    def call(self, inputs, **kwargs):
        m, s = inputs
        epsilon = K.random_normal(shape=K.shape(m))
        S = Multiply()([epsilon, s])
        return m + S


def vae(input_features, isMulticlass, layers, num_cls):
    input = Input(shape=(input_features,))

    x = Dense(1024, input_dim=input_features, activation='relu')(input)
    x = BatchNormalization()(x)
    x = Dropout(0.01)(x)

    if layers > 1:
        x = Dense(768, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)
    if layers > 2:
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)
    if layers > 3:
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)
    if layers > 4:
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)

    m = Dense(units=10)(x)
    s = Dense(units=10)(x)
    x = SamplingLayer()([m, s])
    mu_var = Concatenate(axis=1, name='kl')([m, s])

    if isMulticlass:
        x1 = Dense(num_cls)(x)
        x1 = Activation('softmax', name='cls')(x1)
    else:
        x1 = Dense(1)(x)
        x1 = Activation('sigmoid', name='cls')(x1)

    if layers > 4:
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)
    if layers > 3:
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)
    if layers > 2:
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)
    if layers > 1:
        x = Dense(768, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)
    x = Dense(1024, input_dim=input_features, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.01)(x)

    x2 = Dense(input_features, name='recons')(x)

    model = Model([input], [x1, x2, mu_var])
    return model


def vae_ls(input_features, layers):
    input = Input(shape=(input_features,))

    x = Dense(1024, input_dim=input_features, activation='relu')(input)
    x = BatchNormalization()(x)
    x = Dropout(0.01)(x)

    if layers > 1:
        x = Dense(768, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)
    if layers > 2:
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)
    if layers > 3:
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)
    if layers > 4:
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)

    m = Dense(units=10)(x)
    s = Dense(units=10)(x)
    x = SamplingLayer()([m, s])

    model = Model([input], [x])
    return model


def train(args, model, checkpoint_root, data):
    x_train, x_test, y_train, y_test = data
    optim = tf.keras.optimizers.Adam(learning_rate=args.lr)

    if args.isMulticlass:
        model.compile(loss={'cls': 'categorical_crossentropy', 'recons': 'mse', 'kl': losses.kl_loss(args.beta)},
                      optimizer=optim,
                      metrics={'cls': [tf.keras.metrics.CategoricalAccuracy(), f1_score]})
    else:
        model.compile(loss={'cls': 'binary_crossentropy', 'recons': 'mse', 'kl': losses.kl_loss(args.beta)},
                      optimizer=optim,
                      metrics={'cls': [tf.keras.metrics.BinaryAccuracy(), f1_score]})

    checkpointer = callbacks.ModelCheckpoint(filepath="%s/best-f1.hdf5" % checkpoint_root, verbose=1, mode='max',
                                             monitor='val_cls_f1_score', save_best_only=True)
    csv_logger = CSVLogger('%s/training_set_analysis.csv' % checkpoint_root, separator=',', append=False)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_cls_f1_score', mode='max', patience=200)
    trainCallbacks = [checkpointer, csv_logger, earlyStop]

    model.fit(x=x_train, y=[y_train, x_train, y_train], shuffle=True,
              validation_data=(x_test, [y_test, x_test, y_test]),
              batch_size=args.bs, epochs=args.epochs, callbacks=trainCallbacks)


def test(args, model, data):
    _, x_test, _, y_test = data
    y_test = np.argmax(y_test, axis=-1) if args.isMulticlass else y_test
    y_temp, _, _ = model.predict(x_test[0:args.bs])
    y_pred = np.argmax(y_temp, axis=-1) if args.isMulticlass else np.round(y_temp)
    for idx in range(args.bs, len(x_test), args.bs):
        y_temp, _, _ = model.predict(x_test[idx:idx + args.bs])
        y_pred = np.concatenate((y_pred, np.argmax(y_temp, axis=-1) if args.isMulticlass else np.round(y_temp)), axis=0)
    conf_mat = confusion_matrix(y_test, y_pred)
    return conf_mat


def ls_extract(args, model, data):
    _, x_test, _, y_test = data
    y_test = np.argmax(y_test, axis=-1) if args.isMulticlass else y_test
    ls_temp = model.predict(x_test[0:64])
    ls_pred = ls_temp
    for idx in range(64, len(x_test), 64):
        ls_temp = model.predict(x_test[idx:idx + 64])
        ls_pred = np.concatenate((ls_pred, ls_temp), axis=0)

    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(ls_pred)

    plt.figure(figsize=(16, 10))
    df_subset = pd.DataFrame()
    df_subset['dim 1'] = X_embedded[:, 0]
    df_subset['dim 2'] = X_embedded[:, 1]
    df_subset['y'] = np.array(y_test, dtype=int)

    sns.scatterplot(
        x='dim 1', y='dim 2',
        hue='y',
        palette=sns.color_palette("hls", int(np.max(y_test) + 1)),
        data=df_subset,
        legend="full",
        alpha=0.3
    )

    plt.show()
