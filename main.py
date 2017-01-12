import numpy as np
import graph
from graph import Observation, Edge, Graph
import constant as Constant

def sample(probabilities):
	''' return the index from the list probabilities'''
	indexes = np.arange(len(probabilities))
	probabilities /= np.sum(probabilities) # normalize
	return sum(np.random.choice(indexes, 1, p=probabilities),0)

def metropolis_hastings(iterations, obs, g):
	''' returns one switch settings as induced by MH'''

	# initialize a random sigma (switch settings) 
	sigma = g
	sigma.randomize_switches()

	for i in range(iterations):
		# change sigma by inverting a switch in the graph (sample from q)

		p_star_old_state = p_O_G_sigma(sigma, obs)

		flipped_switch_index = sigma.flip_random_switch()

		p_star_new_state = p_O_G_sigma(sigma, obs)

		alpha = p_star_new_state / p_star_old_state # TODO safeguard against 0
		r = min (1,alpha)
		u = np.random.uniform()

		if u >= r:
			sigma.vertexes[flipped_switch_index].flip_switch_state()

	return sigma.get_sigma()

def complete_prob(s, obs, g, sigmas, probs):
	''' returns p(s|G,O) (full probability) to end up in state s after obs observations. sigmas are
		the sigmas generated from MH, each with probability in probs for the same index '''
	s_prob = 0.0
	for i, sigma in enumerate(sigmas):
		g.set_sigma(sigma)
		sigma_prob = probs[i]
		s_and_O_prob = p_s_O_G_Sigma(s, obs) # s switch state changes with g? (should be fine)
		O_prob = p_O_G_sigma(g, obs)
		s_prob += s_and_O_prob*sigma_prob/O_prob

	return s_prob



def p_s_O_G_Sigma(s, obs):
	''' returns p(s,O|G,sigma) '''
	t = len(obs) - 1
	prob = 0

	# call c(s,t) for all possible directions of s (R,L,O)
	for exit_edge in [Edge.L, Edge.R, Edge.O]:
		s_tuple = (s, exit_edge)
		prob += c(s_tuple, t, obs)

	return prob

def c(s,t, obs):
	''' returns c(s,t) where s is state and t is time step'''
	(v, e) = s # v is vertex reference, e is "source", the value we exited through (R,L,O)

	o = graph.exit_edge(obs[t]) # o is where we exited from our current node according to obs

	if e == Edge.O:
		f = v.get_predecessor(Edge.L)
		g = v.get_predecessor(Edge.R)
		(u, exit_edge_f) = f # DEBUGGING PURPOSES u is node and eu is what (R,L,O) we used to get to v (current node)
		(w, exit_edge_g) = g 
	else:
		f = v.get_predecessor(Edge.O)
		(u, exit_edge_f) = f

	switch_state_L = v.switch_state # true if switch at v is O-L, false if O-R

	# base Case
	if t == 0:
		return 1.0/Constant.N
	# case 1
	elif e == Edge.O and o == Edge.O: 
		return (c(f,t-1, obs)+c(g,t-1,obs))*(1-Constant.p)
	# case 2
	elif e == Edge.O and o != Edge.O:
		return (c(f,t-1, obs)+c(g,t-1, obs))*Constant.p
	# case 3 
	elif e == Edge.L and switch_state_L and o == Edge.L:
		return c(f, t-1,obs)*(1-Constant.p)
	# case 4
	elif e == Edge.L and switch_state_L and o != Edge.L:
		return c(f,t-1,obs)*Constant.p
	# case 5
	elif e == Edge.R and (not switch_state_L) and o == Edge.R:
		return c(f, t-1,obs)*(1-Constant.p)
	# case 6
	elif e == Edge.R and (not switch_state_L) and o != Edge.R:
		return c(f, t-1,obs)*Constant.p
	# case 7
	elif e == Edge.L and (not switch_state_L):
		return 0.0
	# case 8
	elif e == Edge.R and switch_state_L:
		return 0.0

	print "c(s,t) crashed: ", e, o, exit_edge_f, switch_state_L

def p_O_G_sigma(g, obs):
	''' returns p(O|G,sigma) by summing out s from p(s,O|G,sigma) '''
	prob = 0
	for s in g.vertexes:
		prob += p_s_O_G_Sigma(s, obs)

	return prob

def create_distribution(obs, iters, g):
	''' returns distribution of sigma sampled using Metropolis Hastings '''
	sigmas = []
	counts = []

	for i in range(iters):
		# sample new sigma using MH
		new_sigma = metropolis_hastings(Constant.MH_ITERS, obs, g)
		# check if new_sigma has been sampled before
		if new_sigma not in sigmas:
			sigmas.append(new_sigma)
			counts.append(1.0)
		else:
			index = sigmas.index(new_sigma)
			counts[index] += 1.0

	probs = counts / np.sum(counts)
	return sigmas, probs

if __name__ == '__main__':
	g = Graph(Constant.N) 
	g.print_content()
	observations, end_vertex = g.generate_observations()
	print "the last vertex is: ", end_vertex

	sigmas, probs = create_distribution(observations, 100, g)
	print sigmas, probs

	print complete_prob(g.vertexes[0], observations, g, sigmas,probs)
	print complete_prob(g.vertexes[1], observations, g, sigmas,probs)
	print complete_prob(g.vertexes[2], observations, g, sigmas,probs)
	print complete_prob(g.vertexes[3], observations, g, sigmas,probs)






	