import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np
import os
import math
from numpy import random
from PIL import Image

batch_size = 10			# バッチサイズ10
uses_device = 0 #GPUを使用
image_size = 64		# 生成画像のサイズ
neuron_size = 64		# 中間層のサイズ

if uses_device>=0:
	import cupy as cp
	import chainer.cuda
else:
	cp = np

# ベクトルから画像を生成するNN
class DCGAN_Generator_NN(chainer.Chain):

	def __init__(self):
		# 重みデータの初期値を指定する
		w = chainer.initializers.Normal(scale=0.02, dtype=None)
		super(DCGAN_Generator_NN, self).__init__()
		# 全ての層を定義する
		with self.init_scope():
			self.l0 = L.Linear(100, neuron_size * image_size * image_size // 8 // 8,
							   initialW=w)
			self.dc1 = L.Deconvolution2D(neuron_size, neuron_size // 2, 4, 2, 1, initialW=w)
			self.dc2 = L.Deconvolution2D(neuron_size // 2, neuron_size // 4, 4, 2, 1, initialW=w)
			self.dc3 = L.Deconvolution2D(neuron_size // 4, neuron_size // 8, 4, 2, 1, initialW=w)
			self.dc4 = L.Deconvolution2D(neuron_size // 8, 3, 3, 1, 1, initialW=w)
			self.bn0 = L.BatchNormalization(neuron_size * image_size * image_size // 8 // 8)
			self.bn1 = L.BatchNormalization(neuron_size // 2)
			self.bn2 = L.BatchNormalization(neuron_size // 4)
			self.bn3 = L.BatchNormalization(neuron_size // 8)

	def __call__(self, z):
		shape = (len(z), neuron_size, image_size // 8, image_size // 8)
		h = F.reshape(F.relu(self.bn0(self.l0(z))), shape)
		h = F.relu(self.bn1(self.dc1(h)))
		h = F.relu(self.bn2(self.dc2(h)))
		h = F.relu(self.bn3(self.dc3(h)))
		x = F.sigmoid(self.dc4(h))
		return x	# 結果を返すのみ

# 画像を確認するNN
class DCGAN_Discriminator_NN(chainer.Chain):

	def __init__(self):
		# 重みデータの初期値を指定する
		w = chainer.initializers.Normal(scale=0.02, dtype=None)
		super(DCGAN_Discriminator_NN, self).__init__()
		# 全ての層を定義する
		with self.init_scope():
			self.c0_0 = L.Convolution2D(3, neuron_size //  8, 3, 1, 1, initialW=w)
			self.c0_1 = L.Convolution2D(neuron_size //  8, neuron_size // 4, 4, 2, 1, initialW=w)
			self.c1_0 = L.Convolution2D(neuron_size //  4, neuron_size // 4, 3, 1, 1, initialW=w)
			self.c1_1 = L.Convolution2D(neuron_size //  4, neuron_size // 2, 4, 2, 1, initialW=w)
			self.c2_0 = L.Convolution2D(neuron_size //  2, neuron_size // 2, 3, 1, 1, initialW=w)
			self.c2_1 = L.Convolution2D(neuron_size //  2, neuron_size, 4, 2, 1, initialW=w)
			self.c3_0 = L.Convolution2D(neuron_size, neuron_size, 3, 1, 1, initialW=w)
			self.l4 = L.Linear(neuron_size * image_size * image_size // 8 // 8, 1, initialW=w)
			self.bn0_1 = L.BatchNormalization(neuron_size // 4, use_gamma=False)
			self.bn1_0 = L.BatchNormalization(neuron_size // 4, use_gamma=False)
			self.bn1_1 = L.BatchNormalization(neuron_size // 2, use_gamma=False)
			self.bn2_0 = L.BatchNormalization(neuron_size // 2, use_gamma=False)
			self.bn2_1 = L.BatchNormalization(neuron_size, use_gamma=False)
			self.bn3_0 = L.BatchNormalization(neuron_size, use_gamma=False)

	def __call__(self, x):
		h = F.leaky_relu(self.c0_0(x))
		h = F.dropout(F.leaky_relu(self.bn0_1(self.c0_1(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn1_0(self.c1_0(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn1_1(self.c1_1(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn2_0(self.c2_0(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn2_1(self.c2_1(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn3_0(self.c3_0(h))),ratio=0.2)
		return self.l4(h)	# 結果を返すのみ

# カスタムUpdaterのクラス
class DCGANUpdater(training.StandardUpdater):

	def __init__(self, train_iter, optimizer, device):
		super(DCGANUpdater, self).__init__(
			train_iter,
			optimizer,
			device=device
		)

	# 画像認識側の損失関数
	def loss_dis(self, dis, y_fake, y_real):
		batchsize = len(y_fake)
		L1 = F.sum(F.softplus(-y_real)) / batchsize
		L2 = F.sum(F.softplus(y_fake)) / batchsize
		loss = L1 + L2
		return loss

	# 画像生成側の損失関数
	def loss_gen(self, gen, y_fake):
		batchsize = len(y_fake)
		loss = F.sum(F.softplus(-y_fake)) / batchsize
		return loss

	def update_core(self):
		# Iteratorからバッチ分のデータを取得
		batch = self.get_iterator('main').next()
		src = self.converter(batch, self.device)

		# Optimizerを取得
		optimizer_gen = self.get_optimizer('opt_gen')
		optimizer_dis = self.get_optimizer('opt_dis')
		# ニューラルネットワークのモデルを取得
		gen = optimizer_gen.target
		dis = optimizer_dis.target

		# 乱数データを用意
		rnd = random.uniform(-1, 1, (src.shape[0], 100))
		rnd = cp.array(rnd, dtype=cp.float32)

		# 画像を生成して認識と教師データから認識
		x_fake = gen(rnd)		# 乱数からの生成結果
		y_fake = dis(x_fake)	# 乱数から生成したものの認識結果
		y_real = dis(src)		# 教師データからの認識結果

		# ニューラルネットワークを学習
		optimizer_dis.update(self.loss_dis, dis, y_fake, y_real)
		optimizer_gen.update(self.loss_gen, gen, y_fake)


# ニューラルネットワークを作成
model_gen = DCGAN_Generator_NN()
model_dis = DCGAN_Discriminator_NN()

if uses_device>=0:
	#GPU使う
	chainer.cuda.get_device_from_id(0).use()
	chainer.cuda.check_cuda_available()
	#GPU用にデータを変換
	model_gen.to_gpu()
	model_dis.to_gpu()

images = []

fs = os.listdir('imgs/pokemon/full')
for fn in fs:
	# 画像を読み込んで128×128ピクセルにリサイズ
	img = Image.open('imgs/pokemon/full/' + fn).convert('RGB').resize((64, 64))
	# 画素データを0〜1の領域にする
	hpix = cp.array(img, dtype=cp.float32) / 255.0
	hpix = hpix.transpose(2,0,1)
	# 配列に追加
	images.append(hpix)

# 繰り返し条件を作成する
train_iter = iterators.SerialIterator(images, batch_size, shuffle=True)

# 誤差逆伝播法アルゴリズムを選択する
optimizer_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_gen.setup(model_gen)
optimizer_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_dis.setup(model_dis)

# デバイスを選択してTrainerを作成する
updater = DCGANUpdater(train_iter, \
		{'opt_gen':optimizer_gen, 'opt_dis':optimizer_dis}, \
		device=uses_device)
trainer = training.Trainer(updater, (4000, 'epoch'), out="result")
# 学習の進展を表示するようにする
trainer.extend(extensions.ProgressBar())

# 中間結果を保存する
n_save = 0
@chainer.training.make_extension(trigger=(1000, 'epoch'))
def save_model(trainer):
	# NNのデータを保存
	global n_save
	n_save = n_save+1
	chainer.serializers.save_hdf5( 'genchar_gen'+str(n_save)+'.hdf5', model_gen )
	chainer.serializers.save_hdf5( 'genchar_dis'+str(n_save)+'.hdf5', model_dis )
trainer.extend(save_model)

# 機械学習を実行する
trainer.run()

# 学習結果を保存する
chainer.serializers.save_hdf5( 'genchar.hdf5', model_gen )
