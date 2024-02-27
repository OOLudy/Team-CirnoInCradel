import sys, os
sys.path.append(os.pardir)
from SupportFunction.MNIST import load_mnist
from PIL import Image
import numpy as np
# 如果出现下载问题，请删除下载文件
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
normalize=False)
# MNIST 显示
def img_show(img):
	pil_img = Image.fromarray(np.uint8(img))
	pil_img.show()
#img = x_train[1000] #指定图片
#img = img.reshape(28, 28) 把图像的形状变成原来的尺寸
#img_show(img)

