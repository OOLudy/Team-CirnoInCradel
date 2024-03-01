import pickle
import sys, os
sys.path.append(os.pardir)
from SupportFunction.MNIST import load_mnist
from PIL import Image
import numpy as np
from SupportFunction.ActivationFunction import Functions
# 如果出现下载问题，请删除下载文件
# MNIST 显示
def img_show(img):
	pil_img = Image.fromarray(np.uint8(img))
	pil_img.show()
#img = x_train[1000] #指定图片
#img = img.reshape(28, 28) 把图像的形状变成原来的尺寸
#img_show(img)

#层函数
def get_data():
	(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=False,one_hot_label=False)
	return x_test,t_test
def init_network(): #从示例参数保存位置读取参数
	with open("sample_weight.pkl",'rb') as f:
		network = pickle.load(f)
	return network
# 网络结构
def predict (network,x):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']
	a1 = np.dot(x, W1) + b1
	z1 = Functions.sigmoid(a1)
	a2 = np.dot(z1, W2) + b2
	z2 = Functions.sigmoid(a2)
	a3 = np.dot(z2, W3) + b3
	y = Functions.softmax(a3)
	return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
	y = predict(network, x[i])
	p = np.argmax(y) # 获取概率最高的元素的索引
	if p == t[i]:
		accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
