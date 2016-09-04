import pygame
from math import pi
import math
import numpy as np
import random
import operator
import cPickle as pickle

inp_n = 1

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

class MLP(object):
	def __init__(self,hid_layer_info,fnn_inp,rang):
		self.mat = []
		a = np.random.uniform(-1.0*rang,rang,(fnn_inp+1,hid_layer_info[0]))
		self.mat.append(a)
		for i in range(1,len(hid_layer_info)):
			a = np.random.uniform(-1.0*rang,rang,(hid_layer_info[i-1]+1,hid_layer_info[i]))
			self.mat.append(a)
		a = np.random.uniform(-1.0*rang,rang,(hid_layer_info[len(hid_layer_info)-1]+1,fnn_inp))
		self.mat.append(a)

	def chromo2weight(self,arr):
		weight = []
		for i in range(0,len(arr)/5):
			ind = 5*i
			mul = 1.0
			if arr[ind] >= 5:
				mul = -1.0
			w = ((arr[ind+1]*1000.0 + arr[ind+2]*100.0 + arr[ind+3]*10.0 + arr[ind+4])*mul)/10000.0
			weight.append(w)
		weight = np.asarray(weight)
		return weight

	def chromo2mat(self,arr1):
		arr = self.chromo2weight(arr1)
		temp = []
		init = 0
		for i in range(0,len(self.mat)):
			size = self.mat[i].shape[0]*self.mat[i].shape[1]
			print self.mat[i]
			a = []
			for j in range(init,init+size):
				a.append(arr[j])
			a = np.asarray(a)
			init = size
			print a
			a = a.reshape((self.mat[i].shape[0],self.mat[i].shape[1]))
			print a
			temp.append(a)
		temp = np.asarray(temp)
		self.mat = temp

	def feedforward(self,inp):
		l = np.asarray(inp)
		for i in range(0,len(self.mat)):
			l = np.append(l,[1.0])
			l_new = np.dot(l,self.mat[i])
			#l = sigmoid(l_new)
			l = l_new
		return l

fnn = MLP([2,3],inp_n,1.0)

a = np.random.randint(10,size=85)
fnn.chromo2mat(a)
#print fnn.feedforward([30.0,-30.0])
"""
inp = np.random.uniform(-90.0,90.0,inp_n)

for i in range(0,30):
	print inp
	out = fnn.feedforward(inp)
	out = (out*180.0)-90.0
	inp = out	
"""

