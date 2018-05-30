import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np
from PIL import Image

batch_size = 10
uses_device = -1

class NMIST_Conv_NN(chainer.Chain):
    def __init__(self):
        super(NMIST_Conv_NN,self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1,8,ksize=3)
            self.linear1 = L.Linear(1352, 10)

    def __call__(self, x, t=None, train=True):
        h1 = self.conv1(x)
        h2 = F.relu(h1)
        h3 = F.max_pooling_2d(h2,2)
        h4 = self.linear1(h3)

        return F.softmax_cross_entropy(h4,t) if train else F.softmax(h4)


model = NMIST_Conv_NN()

chainer.serializers.load_hdf5('chapt02.hdf5',model)

image = Image.open('sample/chapt02/test/mnist-0.png').convert('L')

pixels = np.asarray(image).astype(np.float32).reshape(1,1,28,28)
pixels = pixels / 255


result = model(pixels,train=False)

for i in range(len(result.data[0])):
    print( str(i) + '\t' + str(result.data[0][i]))
