import numpy as np
import os
import cPickle
import matplotlib
import matplotlib.pyplot as plt

datasets_dir = '/media/datasets/'

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def plot_img(img, title):
    plt.figure()
    plt.imshow(img, interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def cifar10(ntrain=60000,ntest=10000,onehot=True,aug=False):
    data_dir = os.path.join(datasets_dir,'cifar10/')
    fd = open(os.path.join(data_dir,'cifar10_train.pkl'), 'rb')
    dic = cPickle.load(fd)
    fd.close()
    
    trX = dic["data"]
    trY = dic["labels"]

    fd = open(os.path.join(data_dir,'cifar10_test.pkl'))
    dic = cPickle.load(fd)
    fd.close()
    
    teX = dic["data"]
    teY = dic["labels"]

    trX = trX/255.
    teX = teX/255.

    trX = trX.reshape(-1, 3, 32, 32)
    teX = teX.reshape(-1, 3, 32, 32)

    if aug:
        pixelMean = np.mean(trX,axis=0)

        trX -= pixelMean
        teX -= pixelMean

        trX = np.concatenate((trX, trX[:,:,:,::-1]),axis=0)
        trY = np.concatenate((trY, trY),axis=0)

    if onehot:
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX,teX,trY,teY

def mnist(ntrain=60000,ntest=10000,onehot=True):
	data_dir = os.path.join(datasets_dir,'mnist/')
	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = trX/255.
	teX = teX/255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	return trX,teX,trY,teY
