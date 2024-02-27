# coding: utf-8
try:
	import urllib.request
except ImportError:
	raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
	'train_img': 'train-images-idx3-ubyte.gz',
	'train_label': 'train-labels-idx1-ubyte.gz',
	'test_img': 't10k-images-idx3-ubyte.gz',
	'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))  # dirname:返回文件路径 abspath:返回绝对路径 用于不同环境下的代码执行
# 作用就是返回根目录，具体看这个 https://blog.csdn.net/weixin_43641920/article/details/122236711
# __file__ 作用：灵活调用文件位置
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
	file_path = dataset_dir + "/" + file_name  # 将两个字符串拼接在一起，构建一个文件路径
	
	if os.path.exists(file_path):
		# .exists:如果路径 path 存在，返回 True；如果路径 path 不存在或损坏，返回 False。
		return
	
	print("Downloading " + file_name + " ... ")
	urllib.request.urlretrieve(url_base + file_name, file_path)
	# 函数 urllib.request.urlretrieve(url_base + file_name, file_path)的作用是从指定的URL下载文件，
	# 并将文件保存到本地的指定路径中。在这里，url_base + file_name 构成了完整的文件URL，
	# 而 file_path 指定了文件在本地的保存位置。
	print("Done")


def download_mnist():  # 挨个下载这4个文件
	for v in key_file.values():  # 这里：是一个用于字典的方法，它返回字典中所有的值作为一个列表
		_download(v)


def _load_label(file_name):
	file_path = dataset_dir + "/" + file_name  # 将两个字符串拼接在一起，构建一个完整文件路径
	
	print("Converting " + file_name + " to NumPy Array ...")
	with gzip.open(file_path, 'rb') as f:
		labels = np.frombuffer(f.read(), np.uint8, offset=8)
	# 从一个以gzip压缩的二进制文件中读取数据，并将读取的数据解析成NumPy数组
	print("Done")
	
	return labels


def _load_img(file_name):
	file_path = dataset_dir + "/" + file_name
	
	print("Converting " + file_name + " to NumPy Array ...")
	with gzip.open(file_path, 'rb') as f:
		data = np.frombuffer(f.read(), np.uint8, offset=16)
	# 从一个以gzip压缩的二进制文件中读取数据，并将读取的数据解析成NumPy数组
	data = data.reshape(-1, img_size)
	# 行代码的作用是将原始的data数组重新塑造（reshape）成一个新的形状，
	# 其中每个新的子数组都具有img_size个元素。通过使用-1作为参数，
	# 可以确保调整后的数组在总元素数量上与原始数组保持一致。
	print("Done")
	
	return data


def _convert_numpy():  # 创建字典,将训练和测试数据以字典的形式组织起来，方便后续的数据处理和使用
	dataset = {}
	dataset['train_img'] = _load_img(key_file['train_img'])
	dataset['train_label'] = _load_label(key_file['train_label'])
	dataset['test_img'] = _load_img(key_file['test_img'])
	dataset['test_label'] = _load_label(key_file['test_label'])
	
	return dataset


def init_mnist():
	download_mnist()
	dataset = _convert_numpy()
	print("Creating pickle file ...")
	with open(save_file, 'wb') as f:
		pickle.dump(dataset, f, -1)
	print("Done!")


def _change_one_hot_label(X):  # ONE HOT 格式转换（必要时使用
	T = np.zeros((X.size, 10))
	for idx, row in enumerate(T):
		row[X[idx]] = 1
	
	return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
	"""读入MNIST数据集

	Parameters
	----------
	normalize : 将图像的像素值正规化为0.0~1.0
	one_hot_label :
		one_hot_label为True的情况下，标签作为one-hot数组返回
		one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
	flatten : 是否将图像展开为一维数组

	Returns
	-------
	(训练图像, 训练标签), (测试图像, 测试标签)
	"""
	if not os.path.exists(save_file):
		init_mnist()
	
	with open(save_file, 'rb') as f:
		dataset = pickle.load(f)
	
	if normalize:
		for key in ('train_img', 'test_img'):
			dataset[key] = dataset[key].astype(np.float32)
			dataset[key] /= 255.0
	
	if one_hot_label:
		dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
		dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
	
	if not flatten:
		for key in ('train_img', 'test_img'):
			dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
	
	return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
	init_mnist()
