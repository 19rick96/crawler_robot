import pygame
from math import pi
import math
 
pygame.init()
 
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

size = [700, 700]
screen = pygame.display.set_mode(size)
 
pygame.display.set_caption("Example code for the draw module")

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
				self.reward = self.reward + p5[0] - self.old_x
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
d1 = 30
d2 = -30
itr = 0

while not done:

   	clock.tick(30)
        itr = itr + 1
	for event in pygame.event.get(): # User did something
		if event.type == pygame.QUIT: # If user clicked close
			done=True # Flag that we are done so we exit this loop
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_UP:
				d1 = d1 + 3
				print "%f   %d   %d"%(robot1.factor,d1,d2)
			elif event.key == pygame.K_DOWN:
				d1 = d1 - 3
				print "%f   %d   %d"%(robot1.factor,d1,d2)
			elif event.key == pygame.K_LEFT:
				d2 = d2 + 3
				print "%f   %d   %d"%(robot1.factor,d1,d2)
			elif event.key == pygame.K_RIGHT:
				d2 = d2 - 3
				print "%f   %d   %d"%(robot1.factor,d1,d2)
        # Clear the screen and set the screen background
	screen.fill(WHITE)
	pygame.draw.line(screen, BLACK, [0,600],[700,600], 2)
	th1 = d1*0.0174533
	th2 = d2*0.0174533
	
	robot1.draw(th1,th2)
	
	pygame.display.flip()
 
pygame.quit()
