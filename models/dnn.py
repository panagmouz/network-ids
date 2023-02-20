from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras import callbacks
from keras.callbacks import CSVLogger
import numpy as np

from utils.callbacks import f1_score


def dnn(input_features, isMulticlass, layers, num_cls):
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

    if isMulticlass:
        x = Dense(num_cls)(x)
        x = Activation('softmax')(x)
    else:
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)
    model = Model([input], [x])
    return model


def train(args, model, checkpoint_root, data):
    x_train, x_test, y_train, y_test = data
    optim = tf.keras.optimizers.Adam(learning_rate=args.lr)

    if args.isMulticlass:
        model.compile(loss='categorical_crossentropy', optimizer=optim,
                      metrics=[tf.keras.metrics.CategoricalAccuracy(), f1_score])

        checkpointer = callbacks.ModelCheckpoint(filepath="%s/best-f1.hdf5" % checkpoint_root, verbose=1, mode='max',
                                                 monitor='val_f1_score', save_best_only=True)
        csv_logger = CSVLogger('%s/training_set_analysis.csv' % checkpoint_root, separator=',', append=False)
        earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max', patience=50)
        trainCallbacks = [checkpointer, csv_logger, earlyStop]
    else:
        model.compile(loss='binary_crossentropy', optimizer=optim,
                      metrics=[tf.keras.metrics.BinaryAccuracy()])

        checkpointer = callbacks.ModelCheckpoint(filepath="%s/best-acc.hdf5" % checkpoint_root, verbose=1, mode='max',
                                                 monitor='val_binary_accuracy', save_best_only=True)
        csv_logger = CSVLogger('%s/training_set_analysis.csv' % checkpoint_root, separator=',', append=False)
        earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', patience=50)
        trainCallbacks = [checkpointer, csv_logger, earlyStop]

    model.fit(x=x_train, y=y_train, shuffle=True, validation_data=(x_test, y_test),
              batch_size=args.bs, epochs=args.epochs, callbacks=trainCallbacks)


def test(args, model, data):
    _, x_test, _, y_test = data
    y_test = np.argmax(y_test, axis=-1) if args.isMulticlass else y_test
    y_temp = model.predict(x_test[0:args.bs])
    y_pred = np.argmax(y_temp, axis=-1) if args.isMulticlass else np.round(y_temp)
    for idx in range(args.bs, len(x_test), args.bs):
        y_temp = model.predict(x_test[idx:idx + args.bs])
        y_pred = np.concatenate((y_pred, np.argmax(y_temp, axis=-1) if args.isMulticlass else np.round(y_temp)), axis=0)
    conf_mat = confusion_matrix(y_test, y_pred)
    return conf_mat
