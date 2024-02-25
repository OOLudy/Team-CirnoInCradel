import numpy as np
#ReLu
class Functions:
	def __init__(self,RELU,SOFTMAX):
		self.i = RELU
		self.a = SOFTMAX
	def relu(i):
		return np.maximum(0,i)
	def softmax(a):
		exp_a = np.exp(a)
		sum_exp_a = np.sum(exp_a)
		y = exp_a / sum_exp_a
		return y

