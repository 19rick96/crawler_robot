import pygame
from math import pi
import math
import numpy as np
import random
import operator
import cPickle as pickle
import sys

pop = 60
parent_num = 40

outfile = file('chromosomes_robot2.txt','w')

#	Recurrent Neural Network	##########################################################################################

def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def softmax(arr):
	return np.exp(arr)/sum(np.exp(arr))

class RNN(object):
	def __init__(self,hid,th_size,state_size,rang=1.0):
		self.hid = hid
		self.U = np.random.uniform(-1.0*rang,rang,(th_size + state_size,hid))
		self.V = np.random.uniform(-1.0*rang,rang,(hid,state_size))
		self.W = np.random.uniform(-1.0*rang,rang,(hid,hid))
		self.St = np.full((hid),0)

	def numweights(self):
		size = (self.U.shape[0]*self.U.shape[1]) + (self.V.shape[0]*self.V.shape[1]) + (self.W.shape[0]*self.W.shape[1])
		return size

	def feedforward_(self,theta,state):
		xt = np.concatenate((theta,state),axis=0)
		self.St = np.tanh(np.dot(xt,self.U) + np.dot(self.St,self.W))
		ot = softmax(np.dot(self.St,self.V))
		ind = np.argmax(ot)
		for i in range(0,len(ot)):
			if i == ind:
				ot[i] = 1
			else:
				ot[i] = 0
		return ot

	def chromo2weight(self,arr):
		weight = []
		for i in range(0,len(arr)/5):
			ind = 5*i
			mul = 1.0
			if arr[ind] >= 5:
				mul = -1.0
			w = ((arr[ind+1]*1000.0 + arr[ind+2]*100.0 + arr[ind+3]*10.0 + arr[ind+4])*mul)/1000.0
			weight.append(w)
		weight = np.asarray(weight)
		return weight

	def chromo2mat(self,arr1):
		arr = self.chromo2weight(arr1)
		u = arr[:self.U.shape[0]*self.U.shape[1]]
		v = arr[self.U.shape[0]*self.U.shape[1]:(self.U.shape[0]*self.U.shape[1])+(self.V.shape[0]*self.V.shape[1])]
		w = arr[(self.U.shape[0]*self.U.shape[1])+(self.V.shape[0]*self.V.shape[1]):]
		for i in range(0,self.U.shape[0]):
			for j in range(0,self.U.shape[1]):
				self.U[i][j] = u[j + (i*self.U.shape[1])]		
		for i in range(0,self.V.shape[0]):
			for j in range(0,self.V.shape[1]):
				self.V[i][j] = v[j + (i*self.V.shape[1])]
		for i in range(0,self.W.shape[0]):
			for j in range(0,self.W.shape[1]):
				self.W[i][j] = w[j + (i*self.W.shape[1])]
		self.St = np.full((self.hid),0)

rnn = RNN(4,2,9)

def feedforward(inp1,inp2,w):
	inp1 = clamp(inp1,-60.0,90.0)
	inp2 = clamp(inp2,-90.0,90.0)
	th = []
	th.append(inp1)
	th.append(inp2)
	th = np.asarray(th)
	#inp1 = (inp1+90.0)/1.0
	#inp2 = (inp2+90.0)/1.0
	out = rnn.feedforward_(th,w)
	return out

#	GA	#####################################################################################################################

num_weights = rnn.numweights()
chromo_s = 5 * num_weights

#######################################################################################
pygame.init()
 
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

size = [700, 700]
screen = pygame.display.set_mode(size)
 
pygame.display.set_caption("Robot crawler")

done = False
clock = pygame.time.Clock()

class robot(object):
	def __init__(self,L=100.0,W=75.0,width=7.0,L1=75.0,L2=100.0,sx=100.0,sy=600.0):
		self.L = L
		self.W = W
		self.width = width
		self.L1 = L1
		self.L2 = L2
		self.phi = math.atan(W/L)	
		self.sx = sx
		self.sy = sy
		self.factor = 0
		self.reward = 0
		self.old_x = 0
		self.itr = 0

	def reset(self):
		self.itr = 0
		self.old_x = 0
		r = self.reward
		self.reward = 0
		return r

	def find_coord(self,theta1,theta2):
		factor = (math.sqrt((self.L**2)+(self.W**2))*math.sin(self.phi)) + (self.L1*math.sin(theta1)) - (self.L2*math.cos(theta2-theta1))
		f = (self.L2*math.sin(theta2-theta1)) - (self.L1*math.cos(theta1)) - (math.sqrt((self.L**2)+(self.W**2))*math.cos(self.phi))
		factor = factor/f
		self.factor = factor
		theta = math.atan(factor)
		if theta >= 0:
			#print "theta not 0"
			p1 = []
			p2 = []
			p3 = []
			p4 = []
			p5 = []
			p1.append(self.sx)
			p1.append(self.sy)
			p2.append(self.sx + (self.L*math.cos(theta)))
			p2.append(self.sy - (self.L*math.sin(theta)))
			p3.append(self.sx + (math.sqrt((self.L**2)+(self.W**2))*math.cos(theta+self.phi)))
			p3.append(self.sy - (math.sqrt((self.L**2)+(self.W**2))*math.sin(theta+self.phi)))
			p4.append(p3[0] + (self.L1*math.cos(theta1 + theta)))
			p4.append(p3[1] - (self.L1*math.sin(theta1 + theta)))
			p5.append(p4[0] + (self.L2*math.cos((pi/2) + theta2 - theta1 - theta)))
			p5.append(p4[1] + (self.L2*math.sin((pi/2) + theta2 - theta1 - theta)))
			if self.itr != 0:
				self.reward = self.reward + self.old_x - p5[0]
			self.old_x = p5[0]
			self.itr = self.itr + 1
			return p1,p2,p3,p4,p5
		else : 
			#print "theta 0"
			theta = 0
			p1 = []
			p2 = []
			p3 = []
			p4 = []
			p5 = []
			p1.append(self.sx)
			p1.append(self.sy)
			p2.append(self.sx + (self.L*math.cos(theta)))
			p2.append(self.sy - (self.L*math.sin(theta)))
			p3.append(self.sx + (math.sqrt((self.L**2)+(self.W**2))*math.cos(theta+self.phi)))
			p3.append(self.sy - (math.sqrt((self.L**2)+(self.W**2))*math.sin(theta+self.phi)))
			p4.append(p3[0] + (self.L1*math.cos(theta1 + theta)))
			p4.append(p3[1] - (self.L1*math.sin(theta1 + theta)))
			p5.append(p4[0] + (self.L2*math.cos((pi/2) + theta2 - theta1 - theta)))
			p5.append(p4[1] + (self.L2*math.sin((pi/2) + theta2 - theta1 - theta)))
			self.itr = 0
			return p1,p2,p3,p4,p5

	def draw(self,theta1,theta2):
		p1,p2,p3,p4,p5 = self.find_coord(theta1,theta2)
		pygame.draw.line(screen,RED,[int(p1[0]),int(p1[1])],[int(p2[0]),int(p2[1])],int(self.width))
		pygame.draw.line(screen,RED,[int(p2[0]),int(p2[1])],[int(p3[0]),int(p3[1])],int(self.width))
		pygame.draw.line(screen,BLACK,[int(p3[0]),int(p3[1])],[int(p4[0]),int(p4[1])],4)
		pygame.draw.line(screen,BLACK,[int(p4[0]),int(p4[1])],[int(p5[0]),int(p5[1])],4)

robot1 = robot()

def animate(theta1,theta2,w_arr,w=2.0):
	ind = np.argmax(w_arr)
	if ind == 0:
		theta1 = theta1 - w
		theta2 = theta2 - w
	elif ind == 1:
		theta1 = theta1 - w
	elif ind == 2:
		theta1 = theta1 - w
		theta2 = theta2 + w
	elif ind == 3:
		theta2 = theta2 - w
	elif ind == 5:
		theta2 = theta2 + w
	elif ind == 6:
		theta1 = theta1 + w
		theta2 = theta2 - w
	elif ind == 7:
		theta1 = theta1 + w
	else :
		theta1 = theta1 + w
		theta2 = theta2 + w
	clock.tick(60)
	for event in pygame.event.get():
		if event.type == pygame.QUIT: 
			sys.exit(0)
	screen.fill(WHITE)
	pygame.draw.line(screen, BLACK, [0,600],[700,600], 2)
	th1 = theta1*0.0174533
	th2 = theta2*0.0174533
	robot1.draw(th1,th2)
	pygame.display.flip()	
	theta1 = clamp(theta1,-60.0,90.0)
	theta2 = clamp(theta2,-90.0,90.0)
	return theta1,theta2
"""
arr = [5 ,9, 0, 6, 1, 9, 8, 8, 3, 0, 3, 7, 1, 4, 9, 8, 3, 1, 3, 8, 0, 8, 6, 1, 1, 9, 2, 0, 6, 9, 4, 1, 9, 2, 2, 6, 6,
 1, 8, 7, 2, 9, 6, 7, 2, 6, 1, 7, 5, 9, 9, 5, 7, 0, 1, 5, 1, 0, 0, 8, 1, 6, 2, 6, 2, 6, 1, 4, 9, 9, 6, 4, 1, 7,
 3, 2, 6, 4, 7, 8, 9, 1, 9, 8, 4, 9, 4, 7, 4, 0, 1, 6, 3, 7, 6, 5, 2, 9, 9, 0, 6, 3, 3, 7, 9, 1, 2, 7, 3, 6, 4,
 9, 9, 6, 5, 9, 2, 9, 1, 0, 6, 6, 4, 9, 9, 1, 2, 0, 2, 4, 3, 3, 6, 0, 7, 2, 2, 2, 5, 6, 1, 0, 2, 1, 1, 3, 0, 2,
 5, 1, 2, 0, 1, 2, 2, 3, 4, 9, 6, 3, 9, 3, 1, 3, 1, 9, 4, 4, 6, 8, 1, 9, 8, 7, 4, 2, 8, 5, 1, 5, 4, 1, 9, 5, 3,
 0, 8, 4, 4, 7, 7, 6, 6, 4, 9, 6, 3, 7, 7, 0, 9, 8, 0, 7, 4, 8, 4, 6, 1, 0, 7, 8, 1, 6, 7, 9, 6, 8, 0, 1, 9, 3,
 4, 0, 5, 4, 1, 0, 6, 4, 0, 4, 8, 9, 4, 6, 5, 8, 9, 4, 1, 7, 4, 0, 5]

arr = [5 ,5 ,1 ,8 ,8 ,4 ,8 ,4 ,7 ,0 ,3 ,3 ,3 ,6 ,5 ,9 ,3 ,6 ,1 ,9 ,6 ,6 ,2 ,5 ,8 ,8 ,4 ,1 ,5 ,5 ,1 ,6 ,2 ,9 ,0 ,3 ,4,
 9 ,3 ,9 ,0 ,0 ,4 ,2 ,5 ,3 ,0 ,6 ,1 ,8 ,3 ,0 ,1 ,7 ,2 ,5 ,5 ,8 ,9 ,4 ,3 ,1 ,1, 7, 2 ,4 ,0 ,4 ,3 ,2 ,0 ,7 ,9 ,3,
 9 ,7 ,9 ,7 ,8 ,3 ,5 ,2 ,4 ,4 ,8 ,2 ,2 ,3 ,0 ,7 ,9 ,7 ,7 ,5 ,0 ,4 ,8 ,5 ,2 ,8 ,1 ,8 ,2 ,2 ,6 ,7 ,8 ,4 ,3 ,4 ,8,
 8 ,3 ,1 ,2 ,3 ,9 ,3 ,9 ,3 ,1 ,2 ,2 ,1 ,9 ,0 ,5 ,5 ,0 ,5 ,7 ,0 ,9 ,2 ,4 ,9 ,3 ,4 ,8 ,5 ,5 ,1 ,1 ,8 ,6 ,7 ,8 ,3,
 6 ,9 ,9 ,7 ,6 ,5 ,1 ,2 ,5 ,6 ,1 ,5 ,3 ,0 ,9 ,3 ,3 ,6 ,5 ,5 ,4 ,4 ,3 ,1 ,9 ,2 ,0 ,1 ,9 ,8 ,0 ,0 ,4 ,7 ,8 ,4 ,8,
 6 ,7 ,5 ,7 ,2 ,8 ,5 ,9 ,5 ,7 ,5 ,4 ,9 ,5 ,4 ,8 ,1 ,4 ,5 ,2 ,0 ,3 ,4 ,2 ,7]

"""
arr = [3, 9, 9, 7, 0, 2, 8, 6, 7, 3, 3, 7, 3, 3, 1, 4, 2, 1, 4, 5, 6, 5, 4, 9, 7, 0, 9, 1, 9, 3, 3, 8, 1, 9, 6, 8, 5,
 2 ,0 ,4 ,1 ,6 ,3 ,0 ,8 ,3 ,1 ,1 ,5 ,6 ,1 ,3 ,9 ,0 ,6 ,8 ,1 ,7 ,3 ,6 ,1 ,2 ,4 ,0 ,4 ,5 ,9 ,6 ,9 ,4 ,0 ,4 ,8 ,0,
 6 ,9 ,3 ,8 ,5 ,9 ,0 ,4 ,7 ,5 ,5 ,8 ,7 ,4 ,6 ,4 ,9 ,6 ,5, 5, 3, 4, 5, 1, 8, 0, 9, 5, 7, 7, 1, 0, 1, 7, 1, 2, 5,
 4 ,6 ,9 ,4 ,4 ,6 ,2 ,6 ,4 ,6 ,6 ,2 ,5 ,3 ,3, 0 ,1 ,3 ,3 ,2 ,3 ,3 ,6 ,7 ,8 ,8 ,1 ,3 ,4 ,1 ,9 ,2 ,1 ,8 ,5 ,4 ,1,
 6 ,2 ,5 ,5 ,6 ,5 ,2 ,7 ,3 ,3 ,1 ,5 ,5 ,4 ,9 ,1 ,4 ,2 ,3 ,8 ,3 ,4 ,9 ,8 ,1 ,3 ,1 ,6 ,9 ,6 ,0 ,7 ,6 ,0 ,7 ,1 ,8,
 5 ,4 ,3 ,2 ,9 ,1 ,8 ,1, 2 ,9 ,5 ,3 ,0 ,7 ,8 ,5 ,9 ,1 ,7 ,2, 1 ,5 ,5 ,6 ,1 ,0 ,3 ,9 ,5 ,9 ,9 ,2 ,0 ,5 ,8 ,3 ,4,
 2 ,1 ,5 ,4 ,3 ,8 ,4 ,7, 8 ,3 ,1 ,9 ,9 ,0, 9 ,6 ,8 ,1 ,3 ,2 ,6 ,2 ,2 ,5 ,5 ,2 ,1 ,2 ,6 ,3 ,0 ,7 ,5 ,8 ,5 ,2 ,5,
 6 ,6 ,4 ,3 ,0 ,8 ,7 ,4 ,5 ,2 ,0 ,8, 1, 0, 8, 3, 7 ,9 ,6 ,2 ,1 ,1 ,5 ,0 ,1 ,6 ,6 ,5 ,8 ,8 ,3 ,9 ,9 ,0 ,8 ,3 ,4,
 7 ,0 ,6 ,5 ,1 ,5 ,4 ,9 ,5 ,0 ,7 ,2 ,1 ,6 ,3 ,5 ,8 ,9 ,7 ,9 ,7 ,5 ,0 ,1 ,8 ,5 ,8 ,7 ,5 ,5 ,4, 8, 4, 0, 2, 2, 0,
 7 ,6 ,3 ,0 ,1 ,0 ,7 ,9 ,7 ,8 ,0 ,1 ,3 ,1 ,8 ,7 ,9 ,2 ,1 ,4 ,5 ,6 ,0 ,3 ,4 ,4 ,0 ,4 ,8 ,7 ,4, 2, 5, 1, 0, 9, 2,
 8 ,1 ,6 ,8 ,3 ,8 ,0 ,0 ,5 ,9 ,4 ,4 ,2 ,3 ,8 ,1 ,4 ,1 ,2 ,1 ,5 ,3 ,7 ,0 ,1 ,7 ,0 ,1 ,8 ,8 ,9 ,2 ,9 ,4 ,8 ,7 ,4,
 7 ,0 ,5 ,1 ,2 ,5 ,7 ,5 ,2 ,2 ,9 ,4 ,5 ,6 ,3 ,1 ,2 ,6 ,6 ,9, 2, 8, 3, 4, 0, 2, 5, 4, 7, 5, 1, 5, 9, 1, 5, 9, 3,
 8 ,4 ,7 ,1 ,1 ,7 ,4 ,0 ,7 ,1 ,1 ,9 ,1 ,1 ,1, 3, 5, 4, 0, 6, 7, 8, 1, 6, 1, 4, 9, 6, 8, 7, 9, 6, 9, 0, 1, 0]
d1_old = 60
d2_old = -60
w_o = [1,0,0,0,0,0,0,0,0]
fl = 0
clock.tick(60)
screen.fill(WHITE)
pygame.draw.line(screen, BLACK, [0,600],[700,600], 2)
th1 = d1_old*0.0174533
th2 = d2_old*0.0174533
robot1.draw(th1,th2)
pygame.display.flip()
rnn.chromo2mat(arr)
for i in range(0,10000):
	s_new = feedforward(d1_old,d2_old,w_o)
	d1_new,d2_new = animate(d1_old,d2_old,s_new)
	d1_old = d1_new
	d2_old = d2_new
	w_o = s_new

