from __future__ import print_function
import numpy as np
np.random.seed(1234)

from keras.datasets import mnist
from keras.models import Sequential
from keras.callbacks import History 
from keras.regularizers import l2, activity_l2, l1
from keras.layers import Input, merge
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from keras.layers.core import Dense, Dropout, Activation
from k_clayers import NewBatchNormalization as BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from sklearn.metrics import f1_score, accuracy_score
import sklearn
import load
import sys

batch_size = 64
nb_classes = 2
nb_epoch = 200

X_train, X_val, Y_train, Y_val = load.loadData(onehot = False)

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'test samples')

weights = sklearn.utils.compute_class_weight('balanced', [0,1], Y_train)

nFeats = [500]

def model(input):
    net = Dense(nFeats[0], init='he_normal')(input)

    for i in range(5):
        net2 = Dropout(0.2)(net)
        net2 = Dense(nFeats[0], init='he_normal')(net2)
        net2 = BatchNormalization(gamma_regularizer=l2(0.0001))(net2)
        net2 = Activation('relu')(net2)
        net = merge([net, net2], mode='sum')

    net = Dense(1, init='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('sigmoid')(net)
    return net

input = Input(shape=(3,))

model = Model(input=input, output=model(input))

model.summary()


def lr_schedule(epoch):
    if epoch < 10: rate = 0.01
    elif epoch < 80: rate = 0.1
    elif epoch < 120: rate = 0.01
    else: rate = 0.001
    print (rate)
    return rate

sgd = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)

lrate = LearningRateScheduler(lr_schedule)

callbacks = [lrate]

model.compile(loss='binary_crossentropy', optimizer=Nadam(), metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_val, Y_val), shuffle=True, class_weight={0:weights[0], 1:weights[1]})

def predict(x):
    p = model.predict(x)
    p = [0 if x < 0.5 else 1 for x in p]
    return p

print ("Training F1: ", f1_score(Y_train, predict(X_train)))
print ('Val F1:', f1_score(Y_val, predict(X_val)))

X_test, Y_test = load.loadData(onehot = False, split=False, fileName='test.pkl')
print ('Test F1:', f1_score(Y_test, predict(X_test)))
