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
train,test = chainer.datasets.get_mnist(ndim=3)
train_iter = iterators.SerialIterator(train, batch_size, shuffle=True)
test_iter = iterators.SerialIterator(test, 1, repeat=False, shuffle=False)

optimizer = optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device = uses_device)
trainer = training.Trainer(updater,(5,'epoch'), out="result")

trainer.extend(extensions.Evaluator(test_iter, model, device=uses_device))

trainer.extend(extensions.ProgressBar())

# Evaluator
#trainer.extend(extensions.Evaluator(test_iter, model, device=uses_device))

#LogReport
#trainer.extend(extensions.LogReport()))

# PrintReport
#trainer.extend(extensions.PrintReport( entries=['epoch', 'main/loss', 'main/accuracy', 'elapsed_time' ]))
#trainer.extend(extensions.dump_graph(root_name="main/loss", out_name="cg.dot"))
#trainer.run()

#chainer.serializers.save_hdf5('chapt02.hdf5',model)
