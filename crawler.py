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
	def __init__(self,hid_layer_info,fnn_inp,fnn_op,rang=1.0):
		self.mat = []
		a = np.random.uniform(-1.0*rang,rang,(fnn_inp+1,hid_layer_info[0]))
		self.mat.append(a)
		for i in range(1,len(hid_layer_info)):
			a = np.random.uniform(-1.0*rang,rang,(hid_layer_info[i-1]+1,hid_layer_info[i]))
			self.mat.append(a)
		a = np.random.uniform(-1.0*rang,rang,(hid_layer_info[len(hid_layer_info)-1]+1,fnn_op))
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

fnn = MLP([8],4,2)

def feedforward(inp1,inp2,inp3,inp4):
	inp1 = clamp(inp1,-60.0,90.0)
	inp2 = clamp(inp2,-90.0,90.0)
	#inp1 = (inp1+90.0)/1.0
	#inp2 = (inp2+90.0)/1.0
	inp = []
	inp.append(inp1)
	inp.append(inp2)
	inp.append(inp3)
	inp.append(inp4)
	out = fnn.feedforward_(inp)
	return out[0],out[1]

# GA ############################################################
num_weights = fnn.numweights()
chromo_s = 5 * num_weights

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

############################################################################
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

def animate(theta1,theta2,w1,w2,w=2.0):
	if w1>0.5:
		theta1 = theta1 + w
	if w1<0.5:
		theta1 = theta1 - w
	if w2>0.5:
		theta2 = theta2 + w
	if w2<0.5:
		theta2 = theta2 - w
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

def fitness(arr):
	#d1_old = np.random.uniform(-90.0,90.0)
	#d2_old = np.random.uniform(-90.0,90.0)
	d1_old = 60
	d2_old = -60
	w1_o = 0.01
	w2_o = 0.01
	fl = 0
	clock.tick(60)
	screen.fill(WHITE)
	pygame.draw.line(screen, BLACK, [0,600],[700,600], 2)
	th1 = d1_old*0.0174533
	th2 = d2_old*0.0174533
	robot1.draw(th1,th2)
	pygame.display.flip()
	fnn.chromo2mat(arr)
	for i in range(0,350):
		w1,w2 = feedforward(d1_old,d2_old,w1_o,w2_o)
		d1_new,d2_new = animate(d1_old,d2_old,w1,w2)
		d1_old = d1_new
		d2_old = d2_new
		w1_o = w1
		w2_o = w2
	f = robot1.reset()
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

