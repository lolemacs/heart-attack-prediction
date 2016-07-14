import theano
from theano import tensor as T
import numpy as np
import load
from load2 import mnist
from sklearn import datasets
from utils import *
import sys

sys.setrecursionlimit(10000)

trX, teX, trY, teY = load.loadData(onehot = True, poly = 3, prep = 'std')
trX, teX, trY, teY = mnist(onehot = True)

def model(X, drop1, drop2):
    u = dropout(X, drop1)
    u = T.dot(u,w1)
    u = leaky_relu(u, slopes[0])
    u = T.nnet.bn.batch_normalization(u, 1., 0., T.mean(u), T.std(u))
    for i in range(len(wH)):
        t = dropout(u, drop1)
        t = T.dot(t, wH[i])
        t = dropout(clippedK[i], drop1) * leaky_relu(t, slopes[i+1])
        u = t + u
        u = T.nnet.bn.batch_normalization(u, 1., 0., T.mean(u), T.std(u))
    u = dropout(u, drop2)
    u = T.dot(u, w2)
    return T.nnet.softmax(u)

X = T.fmatrix()
Y = T.fmatrix()

nNeurons = 500
nLayers = 5

w1 = init_weights((784, nNeurons))
wH = [init_weights((nNeurons, nNeurons)) for i in range(nLayers)] 
w2 = init_weights((nNeurons, 10))

slopes = theano.shared(floatX(np.ones(nLayers + 1) * 0.25))

K = theano.shared(floatX([.5]*nLayers))
clippedK = clip(K)

noise_py_x = model(X, 0.2, 0.5)
py_x = model(X, 0.0, 0.0)

y_pred = T.argmax(py_x, axis=1)

learnRate = theano.shared(floatX(0.001))

cost = meanClippedCE(noise_py_x, Y)

params = [w1] + wH + [K] + [w2]  + [slopes]
updates = adam(cost, params, learning_rate = learnRate)
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

f = open("results","a")
g = open("trainResults","a")

nEpochs = 400
#timeOut = nEpochs / 2
for i in range(nEpochs):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print "----------------------------------"
    print i
    tr = np.mean(np.argmax(trY, axis=1) == predict(trX))
    g.write("%s\n"%tr)
    print "train acc: ", tr 
    val = np.mean(np.argmax(teY, axis=1) == predict(teX))
    f.write("%s\n"%val)
    print "vld acc: ", val
    print "mean weights: ", np.mean(map(lambda x: x.get_value(), wH))
    print "slopes: ", slopes.get_value()
    print "k's:", clippedK.eval()
    #if ((i + 1) == timeOut):
    #    learnRate.set_value(learnRate.get_value() * floatX(0.5))
    #    timeOut += (nEpochs - timeOut) / 2 
    if ((i + 1) % 50 == 0): learnRate.set_value(learnRate.get_value() * floatX(0.9))
    #if ((i + 1) % 10 == 0): degenRate.set_value(degenRate.get_value() * floatX(3.0))
f.close()
g.close()
