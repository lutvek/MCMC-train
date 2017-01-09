import numpy as np
import random, time, math, itertools
from collections import OrderedDict
import matplotlib.pyplot as plt
from enum import Enum
import constant



class Observation(Enum):
	LO = 1
	RO = 2
	OL = 3
	OR = 4

def observation_str(obs):
	if obs == Observation.LO:
		return "L --> 0"
	elif obs == Observation.RO:
		return "R --> 0"
	elif obs == Observation.OL:
		return "O --> L"
	elif obs == Observation.OR:
		return "O --> R"

class Vertex:

	def __init__(self):
		self.L = None
		self.R = None
		self.O = None
		self.random_switch_state()

	def is_full(self):
		return 	self.L != None and self.R != None and self.O != None
	
	def random_switch_state(self):
		self.switch_state = random.choice([True, False])
	
	def flip_switch_state(self):
		self.switch_state = not self.switch_state	

	def observation(self, source):
		if self.L == source:
			true_obs = Observation(1)
			next = self.O 
		elif self.R == source:
			true_obs = Observation(2)
			next = self.O 
		else:
			if self.switch_state == True:
				true_obs = Observation(3)
				next = self.L
			else:
				true_obs = Observation(4)
				next = self.R
						
					
		if random.uniform(0, 1) < 1-constant.p:
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
	
	def is_valid(self):
		for vertex in self.vertexes:
			if not vertex.is_full():
				return False
		if len(self.vertexes) != self.N:
				return False 		
		return True	

	def generate_vertexes(self):
		self.vertexes = []	
		for _ in range(self.N):
			self.vertexes.append(Vertex())

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
		obs_length = 10
		observations = []		
		i = 1		
		start = random.randint(0, self.N-1)
		sources = [self.vertexes[start].L, self.vertexes[start].R, self.vertexes[start].O]
		source = random.choice(sources)
		next = start
		while i < obs_length:
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
	
	
g = Graph(20)
g.print_content()
observations = g.generate_observations()

					
