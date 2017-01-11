import numpy as np
import random, time, math, itertools
from collections import OrderedDict
import matplotlib.pyplot as plt
from enum import Enum
import constant as Constant



class Observation(Enum):
	LO = 1
	RO = 2
	OL = 3
	OR = 4

class Edge(Enum):
	O=0
	L=1
	R=2

def observation_str(obs):
	if obs == Observation.LO:
		return "L --> 0"
	elif obs == Observation.RO:
		return "R --> 0"
	elif obs == Observation.OL:
		return "O --> L"
	elif obs == Observation.OR:
		return "O --> R"

def exit_edge(obs):
	if obs == Observation.LO or obs == Observation.RO:
		return Edge.O
	elif obs == Observation.OL:
		return Edge.L
	elif obs == Observation.OR:
		return Edge.R

class Vertex:

	def __init__(self,index, graph):
		self.G = graph
		self.L = None
		self.R = None
		self.O = None
		self.index=index
		self.switch_state = random.choice([True, False]) # True denoted O-L

	def get_predecessor(self, incomming_edge):
		''' returns predecessor entering the current vertex in incomming_edge (R,L,O) along 
		with the edge predecessor exited through to get to the current vertex'''
		if incomming_edge == Edge.L:
			predecessor = self.G.vertexes[self.L]
		elif incomming_edge == Edge.R:
			predecessor = self.G.vertexes[self.R]
		else:
			predecessor = self.G.vertexes[self.O]
		return (predecessor, predecessor.neighbor_edge(self))

	def neighbor_edge(self, neighbor):
		''' returs the current vertex edge to neighbor '''
		if self.L == neighbor.index:
			e = Edge.L
		elif self.R == neighbor.index:
			e = Edge.R
		elif self.O == neighbor.index:
			e = Edge.O
		else:
			print "not neighbors"
			e = None # these is not a neighbor
		return e


	def get_exit_connection(to_vertex):
		''' returns the exit connection (L,R,O) for exiting the current
			node towards to_vertex (which is adjacent node) '''
		if self.L==to_vertex.index:
			return Edge.L
		elif self.R==to_vertex.index:
			return Edge.R
		elif self.O==to_vertex.index:
			return Edge.O

	def is_full(self):
		return 	self.L != None and self.R != None and self.O != None
	
	def flip_switch_state(self):
		self.switch_state = not self.switch_state	

	def observation(self, source):
		if self.L == source:
			true_obs = Observation.LO
			next = self.O 
		elif self.R == source:
			true_obs = Observation.RO
			next = self.O 
		else:
			if self.switch_state == True:
				true_obs = Observation.OL
				next = self.L
			else:
				true_obs = Observation.OR
				next = self.R
						
					
		if random.uniform(0, 1) < 1-Constant.p:
			return true_obs, next
		else:
			wrong_obs = [Observation(i) for i in range(1, 4) if Observation(i) != true_obs] 
			return random.choice(wrong_obs), next
			


	def make_connection(self, vertex):
		if self.L is None:
			self.L = vertex
		elif self.R is None:
			self.R = vertex
		elif self.O is None:
			self.O = vertex

	def print_content(self, index):
		print("-----" + str(index) + "-----")
		if self.switch_state == True:
			print("Switch is connection between L and O")
		else: 
			print("Switch is connection between R and O")
		print(	"R is connected to " + str(self.R) +
				"\nL is connected to " + str(self.L) +
				"\nO is connected to " + str(self.O))

class Graph:

	def __init__(self, N):
		if N % 2 == 1:
			print("N has to be even number")
			return
		self.N = N
		self.vertexes = []
		self.create()
	
	def create(self):
		while not self.is_valid():
			self.generate_vertexes()
			self.connect_vertexes()

	def get_sigma(self):
		''' returns sigma as a list with values for how each switch is set,
		1 denoting L-O and 0 denotong R-O'''
		sigmas = []
		for v in self.vertexes:
			switch_setting = v.switch_state
			if switch_setting: # L-O
				sigmas.append(1)
			else:
				sigmas.append(0)
		return sigmas

	def set_sigma(self, sigma):
		''' sets the switches to the binary list in sigma where 1 denotes L-O
		and 0 denotes R-0'''
		if len(sigma) != len(self.vertexes):
			print "ERROR in set_sigma: sigma must be of the same length as the number of vertexes"
			return 
		for i,v in enumerate(self.vertexes):
			if v.switch_state != bool(sigma[i]): 
				v.flip_switch_state()
	
	def is_valid(self):
		for vertex in self.vertexes:
			if not vertex.is_full():
				return False
		if len(self.vertexes) != self.N:
				return False 		
		return True

	def randomize_switches(self):
		for vertex in self.vertexes:
			if random.randint(0,1) == 0:
				vertex.flip_switch_state()

	def flip_random_switch(self):
		v_index = random.randint(Constant.N-1)
		self.vertexes[v_index].flip_switch_state()
		return v_index

	def generate_vertexes(self):
		self.vertexes = []	
		for i in range(self.N):
			self.vertexes.append(Vertex(i,self))

	def connect_vertexes(self):
		for i in range(self.N-1):
			seen = []
			tries = 0
			while not self.vertexes[i].is_full():		
				z = random.randint(i+1, self.N-1)
				if (not z in seen) and not self.vertexes[z].is_full():
					seen.append(z)
					self.vertexes[z].make_connection(i)
					self.vertexes[i].make_connection(z)
				elif tries > self.N:
					return
				tries += 1
	
	def generate_observations(self):
		observations = []		
		i = 1		
		start = random.randint(0, self.N-1)
		sources = [self.vertexes[start].L, self.vertexes[start].R, self.vertexes[start].O]
		source = random.choice(sources)
		next = start
		while i < Constant.OBS_LEN:
			obs, new_next = self.vertexes[next].observation(source)
			source = next
			next = new_next
			observations.append(obs)
			i += 1
			print("Got observation " + observation_str(obs) + " go to next " + str(next))
		return observations

	def print_content(self):
		for i in range(self.N):
			self.vertexes[i].print_content(i)
	


					
