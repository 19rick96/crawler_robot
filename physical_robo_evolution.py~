from math import pi
import math
import numpy as np
import random
import operator
import cPickle as pickle
import sys
import serial
import time
import cv2

ser = serial.Serial("/dev/ttyACM0",9600)

#initialize camera
cap = cv2.VideoCapture(1)
cv2.namedWindow('camera')

#green for detecting forward movement of robot
lower_green = np.array([40,55,0])
upper_green = np.array([75,200,255])

#communicate with arduino(servo angles)
def send_msg(ang1,ang2):
	values = bytearray([ang1,ang2])
	ser.write(values)
	time.sleep(0.007)

# neural network###############################################
def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n

class MLP(object):
	def __init__(self,hid_layer_info,fnn_inp,rang=1.0):
		self.mat = []
		a = np.random.uniform(-1.0*rang,rang,(fnn_inp+1,hid_layer_info[0]))
		self.mat.append(a)
		for i in range(1,len(hid_layer_info)):
			a = np.random.uniform(-1.0*rang,rang,(hid_layer_info[i-1]+1,hid_layer_info[i]))
			self.mat.append(a)
		a = np.random.uniform(-1.0*rang,rang,(hid_layer_info[len(hid_layer_info)-1]+1,fnn_inp))
		self.mat.append(a)
	
	def numweights(self):
		size = 0
		for i in range(0,len(self.mat)):
			s = self.mat[i].shape[0]*self.mat[i].shape[1]
			size = size + s
		return size		

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
			a = []
			for j in range(init,init+size):
				a.append(arr[j])
			a = np.asarray(a)
			init = size
			a = a.reshape((self.mat[i].shape[0],self.mat[i].shape[1]))
			temp.append(a)
		temp = np.asarray(temp)
		self.mat = temp

	def feedforward_(self,inp):
		l = np.asarray(inp)
		for i in range(0,len(self.mat)):
			l = np.append(l,[1.0])
			l_new = np.dot(l,self.mat[i])
			l = sigmoid(l_new)
			#l = l_new
		return l

fnn = MLP([5,4],2)

def feedforward(inp1,inp2):
	inp1 = clamp(inp1,-60.0,90.0)
	inp2 = clamp(inp2,-90.0,90.0)
	inp = []
	inp.append(inp1)
	inp.append(inp2)
	out = fnn.feedforward_(inp)
	return out[0],out[1]

# GA parameters
num_weights = fnn.numweights()
chromo_s = 5 * num_weights
pop = 60
parent_num = 40

outfile = file('chromosomes_robot_ph.txt','w')

# GA ############################################################
def mutate_weights(arr,mut):
	for j in range(0,mut):
		i = random.randint(0,len(arr)-1)
		a = random.randint(0,9)
		arr[i] = a
	return arr

def n_point_crossover(n,arr1,arr2):
	points = random.sample(range(1,len(arr1)-1),n)
	child = []
	init = 0
	points = sorted(points)
	for i in range(0,len(points)):
		ch = random.randint(0,1)
		if ch == 0:
			for j in range(init,points[i]):
				child.append(arr1[j])
		else :
			for j in range(init,points[i]):
				child.append(arr2[j])
		init = points[i]
	ch = random.randint(0,1)
	if ch == 0:
		for j in range(init,len(arr1)):
			child.append(arr1[j])
	else :
		for j in range(init,len(arr2)):
			child.append(arr2[j])
	child = np.asarray(child)
	return child

def form_mating_pool(parent_pop,population,fit_arr):
	# stochastic universal sampling
	segments = []
	total = 0
	for i in range(0,len(population)):
		total = total + fit_arr[i]
		segments.append(total)
	segments = np.asarray(segments)
	dist = total/parent_pop
	pointer = random.uniform(0.0,dist)
	parents = []
	for i in range(0,parent_pop):
		ind = 0
		par = pointer + i*dist
		for j in range(1,len(segments)):
			if par<segments[j]:
				ind = j
				break
		parents.append(population[ind])
	parents = np.asarray(parents)
	return parents

def animate(theta1,theta2,w1,w2,w=2.0):
	if w1>0.5:
		theta1 = theta1 + w
	if w1<0.5:
		theta1 = theta1 - w
	if w2>0.5:
		theta2 = theta2 + w
	if w2<0.5:
		theta2 = theta2 - w

	send_msg(90 + theta1,90 + theta2)	
	theta1 = clamp(theta1,-60.0,90.0)
	theta2 = clamp(theta2,-90.0,90.0)
	return theta1,theta2

def fitness(arr):

	d_initial = 0
	d_final = 0
	_, frame = cap.read()
	img  = frame.copy()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower_green, upper_green)
	#mask = cv2.erode(mask, None, iterations=2)
	#mask = cv2.dilate(mask, None, iterations=2)
	cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None
	if len(cnts) > 0:
		c = sorted(cnts, key=cv2.contourArea,reverse=True)
		if len(c)<2:
			return -1
		else:
			((x1,y1),radius1) = cv2.minEnclosingCircle(c[0])
			((x2,y2),radius2) = cv2.minEnclosingCircle(c[1])
			cv2.circle(img, (int(x1), int(y1)), int(radius1),(0, 255, 255), 2)
			cv2.circle(img, (int(x2), int(y2)), int(radius2),(0, 255, 255), 2)
			d_initial = abs(x2-x1)

	cv2.imshow('camera',img)

	d1_old = 60
	d2_old = -60
	send_msg(90 + d1_old,90 + d2_old)
	time.sleep(0.2)
	fnn.chromo2mat(arr)
	for i in range(0,350):
		_, frame = cap.read()
		img  = frame.copy()
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, lower_green, upper_green)
		cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
		center = None
		if len(cnts) > 0:
			c = sorted(cnts, key=cv2.contourArea,reverse=True)
			if len(c)<2:
				asd = 1
			else:
				((x1,y1),radius1) = cv2.minEnclosingCircle(c[0])
				((x2,y2),radius2) = cv2.minEnclosingCircle(c[1])
				cv2.circle(img, (int(x1), int(y1)), int(radius1),(0, 255, 255), 2)
				cv2.circle(img, (int(x2), int(y2)), int(radius2),(0, 255, 255), 2)
		cv2.imshow('camera',img)

		w1,w2 = feedforward(d1_old,d2_old)
		d1_new,d2_new = animate(d1_old,d2_old,w1,w2)
		d1_old = d1_new
		d2_old = d2_new

	_, frame = cap.read()
	img  = frame.copy()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower_green, upper_green)
	#mask = cv2.erode(mask, None, iterations=2)
	#mask = cv2.dilate(mask, None, iterations=2)
	cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None
	if len(cnts) > 0:
		c = sorted(cnts, key=cv2.contourArea,reverse=True)
		if len(c)<2:
			return -1
		else:
			((x1,y1),radius1) = cv2.minEnclosingCircle(c[0])
			((x2,y2),radius2) = cv2.minEnclosingCircle(c[1])
			cv2.circle(img, (int(x1), int(y1)), int(radius1),(0, 255, 255), 2)
			cv2.circle(img, (int(x2), int(y2)), int(radius2),(0, 255, 255), 2)
			d_final = abs(x2-x1)

	cv2.imshow('camera',img)

	f = d_initial - d_final
	return f

population = []
for i in range(0,pop):
	a = np.random.randint(10,size=chromo_s)
	population.append(a)

fitness_arr = []

for i in range(0,len(population)):
	f = fitness(population[i])
	fitness_arr.append(f)
	print "initial  %d    %f"%(i+1,f)
print fitness_arr

children = []
parents = form_mating_pool(parent_num,population,fitness_arr)
for j in range(0,2):
	random.shuffle(parents)
	for i in range(0,len(parents)/2):
		c1 = mutate_weights(parents[2*i],13)
		c2 = mutate_weights(parents[(2*i)+1],13)
		c3 = n_point_crossover(4,c1,c2)
		children.append(c3)

generations = 50
gen_itr = 0

while gen_itr<generations:
	gen_itr = gen_itr + 1
	fit_child_arr = []
	for i in range(0,len(children)):
		f = fitness(children[i])
		fit_child_arr.append(f)
		print "%d   gen  children  %d   %f"%(gen_itr,i+1,f)
	temp = []
	for i in range(0,len(population)):
		elem = []
		elem.append(population[i])
		elem.append(fitness_arr[i])
		temp.append(elem)
	for i in range(0,len(children)):
		elem = []
		elem.append(children[i])
		elem.append(fit_child_arr[i])
		temp.append(elem)
	#elitist approach
	temp = sorted(temp,key=operator.itemgetter(1),reverse=True)
	print "%d     %f"%(gen_itr,temp[0][1])
	print temp[0][0]
	best = np.asarray(temp[0][0])
	np.savetxt(outfile,best,fmt='%-7.4f')
	population = []
	fitness_arr = []
	for i in range(0,pop):
		population.append(temp[i][0])
		fitness_arr.append(temp[i][1])		
	children = []
	parents = form_mating_pool(parent_num,population,fitness_arr)
	for j in range(0,2):
		random.shuffle(parents)
		for i in range(0,len(parents)/2):
			c1 = mutate_weights(parents[2*i],13)
			c2 = mutate_weights(parents[(2*i)+1],13)
			c3 = n_point_crossover(4,c1,c2)
			children.append(c3)

