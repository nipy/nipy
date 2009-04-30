"""
Routines for Matching of a graph to a cloud of points/tree structures through
Bayesian networks (Belief propagation) algorithms 

Author: Bertrand Thirion , 2006-2008.

Comment (2009/03/24)
"""

import numpy as np
import nipy.neurospin.graph.graph as fg
from nipy.neurospin.eda.dimension_reduction import Euclidian_distance



def _HDistance(X,Y):
	"""
	Computation of the hausdorff distances between the elements of the lists X
	and Y
	"""
	D = []
	for x in X:
		for y in Y:
			ED = Euclidian_distance(x,y)
			D.append (np.maximum(ED.min(0).max(),ED.min(1).max()))
	D = np.array(D)
	D = np.reshape(D,(len(X),len(Y)))
	return D

def BPmatch(c1,c2,graph,dmax):
	belief = fg.graph_bpmatch(c1,c2,graph,dmax)
	i,j = np.where(belief);
	k = belief[i,j]
	return i,j,k

def match_trivial(c1, c2, dmax, eps = 1.e-12 ):
	"""
	Matching the rows of c1 to those of c2 based on their relative positions
	"""
	s1 = np.size(c1,1)
	s2 = np.size(c2,1)
	sqs = 2*dmax*dmax
	
	# make a prior
	D = Euclidian_distance(c1,c2)
	W = np.exp(-D*D/sqs);
	W = W*(D<3*dmax);
	sW = np.sum(W,1)
	if np.sum(sW)<eps:
		return np.array([]),np.array([]),np.array([])
	W = np.transpose(np.transpose(W)/np.maximum(eps,np.sum(W,1)))
	
	i,j = np.where(W);
	k = W[i,j]
	return i,j,k

	
def BPmatch_slow(c1, c2, graph, dmax, imax= 20, eps = 1.e-12 ):
	"""
	Matching the rows of c1 to those of c2 based on their relative positions
	graph is a matrix that yields a graph structure on the rows of c1
	dmax is measure of the distance decay between points and correspondences
	for algorithmic details, see
	Thirion et al, MMBIA 2006
	"""
	s1 = np.size(c1,1)
	s2 = np.size(c2,1)
	sqs = 2*dmax*dmax
	
	# make a prior
	D = Euclidian_distance(np.transpose(c1),np.transpose(c2))
	W = np.exp(-D*D/sqs);
	W = W*(D<3*dmax);
	sW = np.sum(W,1)
	if np.sum(sW)<eps:
		return np.array([]),np.array([]),np.array([])
	W = np.transpose(np.transpose(W)/np.maximum(eps,np.sum(W,1)))
		
	# make transition matrices
	graph = graph - np.diag(np.diag(graph))
	i,j = np.where(graph)
	if np.size(i)==0:
		i,j = np.where(W);
		k = W[i,j]
		return i,j,k
	E = np.size(i)

	# the variable tag is used to reduce the computation load...
	Mdx = np.zeros((s1,s1))
	Mdx[i,j] = np.arange(E)
	tag = Mdx[j,i]
	#
	
	if E>0 :
		T = []
		for e in range(E):
			t = np.zeros((s2,s2)).astype('d')
			u = c1[:,j[e]]-c1[:,i[e]]
			for k in range(s2):
				for l in range(s2):
					v = (c2[:,l]-c2[:,k]-u)
					nuv = np.sum(v*v)
					t[k,l] = np.exp(-nuv/sqs)
			aux = np.reshape(np.sum(t,1),(s2,1))
			nt = np.transpose(np.repeat(aux,s2,1))
			
			t = t/nt
			T.append(t)
		
		# init the messages
		M = np.zeros((E,s2)).astype('d')
		Pm = W[i,:]
		B = W.copy()
		#print i,j
		
		now = s1 #np.sqrt(np.sum(np.sum(W*W)))
		for iter in range(imax):
			Bold = B.copy()
			B = W.copy()
			
			# compute the messages
			for e in range(E):
				t = T[e]
				plop = np.dot(Pm[e,:],t)
				M[e,:] = plop

			sM = np.sum(M,1)
			sM = np.maximum(sM,eps)
			M = np.transpose(np.transpose(M)/sM)
						
			# update the beliefs
			# B[j,:] = B[j,:]*M
			for e in range(E):
				B[j[e],:] = B[j[e],:]*M[e,:]
			
			B = np.transpose(np.transpose(B)/np.maximum(eps,np.sum(B,1)))
			dB = np.sqrt(np.sum(np.sum(B-Bold)**2))
			if dB<eps*now:
				 # print dB/now,iter
				 break
						
			#prepare the next message
			for e in range(E):
				me = np.maximum(eps,M[tag[e],:])
				Pm[e,:] = B[i[e],:]/me
	else:
		B=W

	
	B = np.transpose(np.transpose(B)/np.maximum(eps,np.sum(B,1)))
	i,j = np.where(B);
	k = B[i,j]
	return i,j,k



def BPmatch_slow_asym(c1, c2, G1, G2, dmax):
	"""
	New version which makes the differences between ascending
	and descending links
	- c1 and c2 are arrays of shape (n1,d) and (n2,d) that represent
	features or coordinates,
	where n1 and n2 are the number of things to be put in correpondence
	and d is the common dim
	- G1 and G2 are corresponding graphs (forests in fff sense)
	- dmax is  a typical distance to compare positions
	"""
	if G1.V != c1.shape[0]:
		raise ValueError, "incompatible dimension for G1 and c1"

	if G2.V != c2.shape[0]:
		raise ValueError, "incompatible dimension for G2 and c2"

	eps = 1.e-7
	sqs = 2*dmax*dmax
	ofweight = np.exp(-0.5)

	# get the distances
	D = Euclidian_distance(c1,c2)
	W = np.exp(-D*D/sqs);
	
	# find the leaves of the graphs	
	sl1 = G1.isleaf().astype('bool')
	sl2 = G2.isleaf().astype('bool')

	# cancel the weigts of non-leaves
	for i in np.where(sl1==0)[0]:
		W[i,:]=1 #!!!!	
	for i in np.where(sl1)[0]:
		W[i,sl2==0]=0

	# normalize the weights
	sW = np.sum(W,1)+ofweight
	W = np.transpose(np.transpose(W)/sW)
	W0 = W.copy()
	
	# get the different trees in each graph
	u1 = G1.cc()
	u2 = G2.cc()
	nt1 = u1.max()+1
	nt2 = u2.max()+1

	# get the tree-summed weights
	tW = np.zeros((G1.V,nt2))
	for i2 in range(nt2):
		tW[:,i2] = np.sum(W[:,u2==i2],1)
	
	# run the algo for each pair of tree
	for i1 in range(nt1):
		if np.sum(u1==i1)>1:
			g1 = G1.subforest(u1==i1)
			for i2 in range(nt2):
				if np.sum(u2==i2)>1:		
					g2 = G2.subforest(u2==i2)
					rW = W[u1==i1,:][:,u2==i2]
					rW = np.transpose(np.transpose(rW)/tW[u1==i1,i2])
					rW = _MP_algo_(g1,g2,rW,c1[u1==i1,:],c2[u2==i2],sqs)
					q = 0
					for j in np.where(u1==i1)[0]:
						W[j,u2==i2] = rW[q,:]*tW[j,i2]
						q = q+1
					
	# cancel the weigts of non-leaves
	W[sl1==0,:]=0
	W[:,sl2==0]=0

	W[W<1.e-4]=0
	i,j = np.where(W)
	k = W[i,j]
	return i,j,k


def BPmatch_slow_asym_dev(c1, c2, G1, G2, dmax):
	"""
	New version which makes the differences between ascending
	and descending links
	INPUT:
	- c1 and c2 are arrays of shape (n1,d) and (n2,d) that represent
	features or coordinates,
	where n1 and n2 are the number of things to be put in correpondence
	and d is the common dim
	- G1 and G2 are corresponding graphs (forests in fff sense)
	- dmax is  a typical distance to compare positions
	OUTPUT:
	- (i,j,k): sparse model of the probabilistic relationships,
	where k is the probability that i is associated with j
	"""
	if G1.V != c1.shape[0]:
		raise ValueError, "incompatible dimension for G1 and c1"

	if G2.V != c2.shape[0]:
		raise ValueError, "incompatible dimension for G2 and c2"

	eps = 1.e-7
	sqs = 2*dmax*dmax
	ofweight = np.exp(-0.5)

	# get the distances
	D = Euclidian_distance(c1,c2)
	W = np.exp(-D*D/sqs)

	# add an extra default value and normalize
	W = np.hstack((W,ofweight*np.ones((G1.V,1))))
	W = np.transpose(np.transpose(W)/np.sum(W,1))

	W = _MP_algo_dev_(G1,G2,W,c1,c2,sqs)
	W = W[:,:-1]
	
	# find the leaves of the graphs	
	sl1 = G1.isleaf().astype('bool')
	sl2 = G2.isleaf().astype('bool')
	# cancel the weigts of non-leaves
	W[sl1==0,:]=0
	W[:,sl2==0]=0

	W[W<1.e-4]=0
	i,j = np.where(W)
	k = W[i,j]
	return i,j,k



def _son_translation_(G,c):
	"""
	Given a forest strcuture G and a set of coordinates c,
	provides the coorsdinate difference ('translation')
	associated with each descending link
	"""
	v = np.zeros((G.E,c.shape[1]))
	for e in range(G.E):
		if G.weights[e]<0:
			ip = G.edges[e,0]
			target = G.edges[e,1]
			v[e,:] = c[target,:]-c[ip,:]		
	return v


def singles(G):
	singles = np.ones(G.V)
	singles[G.edges]=0
	return singles


def _MP_algo_(G1,G2,W,c1,c2,sqs,imax= 100, eps = 1.e-12 ):

	eps = eps
	#get the graph structure
	i1 = G1.get_edges()[:,0]
	j1 = G1.get_edges()[:,1]
	k1 = G1.get_weights()
	i2 = G2.get_edges()[:,0]
	j2 = G2.get_edges()[:,1]
	k2 = G2.get_weights()
	E1 = G1.E
	E2 = G2.E

	# define vectors related to descending links v1,v2
	v1 = _son_translation_(G1,c1)
	v2 = _son_translation_(G2,c2)	

	# the variable tag is used to reduce the computation load...
	if E1>0:
		tag = G1.converse_edge()

		# make transition matrices
		T = []
		for e in range(E1):
			if k1[e]>0:
				# ascending links
				t = eps*np.eye(G2.V)
				
				for f in range(E2):
					if k2[f]<0:
						du = v1[tag[e],:]-v2[f,:]
						nuv = np.sum(du**2)
						t[j2[f],i2[f]] = np.exp(-nuv/sqs)
						
			if k1[e]<0:
				#descending links
				t = eps*np.eye(G2.V)
				
				for f in range(E2):
					if k2[f]<0:
						du = v1[e,:]-v2[f,:]
						nuv = np.sum(du**2)
						t[i2[f],j2[f]] = np.exp(-nuv/sqs)
				
			t = np.transpose(np.transpose(t)/np.maximum(eps,np.sum(t,1)))
			
			T.append(t)

		# the BP algo itself
		# init the messages
		M = np.zeros((E1,G2.V)).astype('d')
		Pm = W[i1,:]
		B = W.copy()
		
		now = float(G1.V) 
		for iter in range(imax):
			Bold = B.copy()
			B = W.copy()
			
			# compute the messages
			for e in range(E1):
				t = T[e]
				M[e,:] = np.dot(Pm[e,:],t)

			sM = np.sum(M,1)
			sM = np.maximum(sM,eps)
			M = np.transpose(np.transpose(M)/sM)
						
			# update the beliefs
			for e in range(E1):
				B[j1[e],:] = B[j1[e],:]*M[e,:]

			B = np.transpose(np.transpose(B)/np.maximum(eps,np.sum(B,1)))
			dB = np.sqrt(((B-Bold)**2).sum())
			if dB<eps*now:
				 # print dB/now,iter
				 break
						
			#prepare the next message
			for e in range(E1):
				me = np.maximum(eps,M[tag[e],:])
				Pm[e,:] = B[i1[e],:]/me
	else:
		B=W

	B = np.transpose(np.transpose(B)/np.maximum(eps,np.sum(B,1)))
	
	return B


def _MP_algo_dev_(G1,G2,W,c1,c2,sqs,imax= 100, eta = 1.e-6 ):
	"""
	W=_MP_algo_dev_(G1,G2,W,c1,c2,sqs,imax= 100, eta = 1.e-6 )
	Internal part of the graph matching procedure.
	Not to be read by normal people.
	INPUT:
	- c1 and c2 are arrays of shape (n1,d) and (n2,d) that represent
	features or coordinates,
	where n1 and n2 are the number of things to be put in correpondence
	and d is the common dim
	- G1 and G2 are corresponding graphs (forests in fff sense)
	- W is the  initial correpondence matrix
	- dmax is  a typical distance to compare positions
	- imax = 100: maximal number of iterations
	(in practice, this number is the diameter of G1)
	- eta = 1.e-6 a constant <<1 that avoids inconsistencies
	OUTPUT:
	- W updated correspondence matrix
	"""
	eps = 1.e-12
	#get the graph structure
	i1 = G1.get_edges()[:,0]
	j1 = G1.get_edges()[:,1]
	k1 = G1.get_weights()
	i2 = G2.get_edges()[:,0]
	j2 = G2.get_edges()[:,1]
	k2 = G2.get_weights()
	E1 = G1.E
	E2 = G2.E

	# define vectors related to descending links v1,v2
	v1 = _son_translation_(G1,c1)
	v2 = _son_translation_(G2,c2)	

	# the variable tag is used to reduce the computation load...
	if E1<1:
		return W

	tag = G1.converse_edge().astype('i')
	
	# make transition matrices
	T = []
	for e in range(E1):
		if k1[e]>0:
			# ascending links
			t = eta*np.ones((G2.V+1,G2.V+1))
			for f in range(E2):
				if k2[f]<0:
					#du = v1[tag[e],:]-v2[f,:]
					#nuv = np.sum(du**2)
					t[j2[f],i2[f]] = 1
						
		if k1[e]<0:
			#descending links
			t = eta*np.ones((G2.V+1,G2.V+1))
			for f in range(E2):
				if k2[f]<0:
					du = v1[e,:]-v2[f,:]
					nuv = np.sum(du**2)
					t[i2[f],j2[f]] = np.exp(-nuv/sqs)
				
		t = np.transpose(np.transpose(t)/np.maximum(eps,np.sum(t,1)))
		T.append(t)

	# the BP algo itself
	# init the messages
	M = np.zeros((E1,G2.V+1)).astype('d')
	Pm = W[i1,:]
	B = W.copy()
	now = float(G1.V) 
		
	for iter in range(imax):
		Bold = B.copy()
		B = W.copy()
					
		# compute the messages
		for e in range(E1):
			t = T[e]
			M[e,:] = np.dot(Pm[e,:],t)

		sM = np.sum(M,1)
		sM = np.maximum(sM,eps)
		M = np.transpose(np.transpose(M)/sM)
						
		# update the beliefs
		for e in range(E1):
			B[j1[e],:] = B[j1[e],:]*M[e,:]

		B = np.transpose(np.transpose(B)/np.maximum(eps,np.sum(B,1)))
		dB = np.sqrt(((B-Bold)**2).sum())
		if dB<eps*now:
			# print dB/now,iter
			break

		#prepare the next message
		for e in range(E1):
			me = np.maximum(eps,M[tag[e],:])
			Pm[e,:] = B[i1[e],:]/me

	B = np.transpose(np.transpose(B)/np.maximum(eps,np.sum(B,1)))	
	return B
	
###--------------------------------------------------------------
###-------------- Some -extremely coarse- tests -----------------
###--------------------------------------------------------------

def _testmatch_():
	c1 = np.array([[0],[1],[2]])
	c2 = c1+0.7
	adjacency = np.ones((3,3))-np.eye(3)
	adjacency[2,0] = 0
	adjacency[0,2] = 0
	dmax = 1.4
	i,j,k = BPmatch(c1, c2, adjacency, dmax)
	GM = np.zeros((3,3))
	GM[i,j]=k
	print GM
	i,j,k = BPmatch_slow(np.transpose(c1), np.transpose(c2), adjacency, dmax)
	GM = np.zeros((3,3))
	GM[i,j]=k
	print GM


def _testmatch_1_():
	"""
	Test of a 2-dimensional case, with 2 triangles with an ambiguous
	correspondence.
	To show the effect of loopy vs non-loopy graphs
	"""
	b = 0.5
	a = np.sqrt(3.)/2
	c1 = np.array([[a,b],[-a,b],[0,-1]])
	c2 = np.array([[0,1],[-a,-b],[a,-b]])
	graph = np.ones((3,3))-np.eye(3)
	dmax = 1.0
	#i,j,k = BPmatch_slow(np.transpose(c1), np.transpose(c2), graph, dmax)
	i,j,k = BPmatch(c1, c2, graph, dmax)
	GM = np.zeros((3,3))
	GM[i,j]=k
	print GM
	graph[2,1] = 0
	graph[1,2] = 0
	
	#i,j,k = BPmatch_slow(np.transpose(c1), np.transpose(c2), graph, dmax)
	i,j,k = BPmatch(c1, c2, graph, dmax)
	GM = np.zeros((3,3))
	GM[i,j]=k
	print GM
	#i,j,k = BPmatch_slow(np.transpose(c2), np.transpose(c1), graph, dmax)
	i,j,k = BPmatch(c2, c1, graph, dmax)
	GM = np.zeros((3,3))
	GM[i,j]=k
	print GM


def _ya_testmatch_():
	"""
	test function the forest matching algorithm
	basically this creates two graphs and associated 
	"""
	c1 = np.array([[0],[-1],[1],[0.5],[1.5]])
	parents = np.array([0, 0, 0, 2, 2])
	g1 = fg.Forest(5,parents)

	c2 = np.array([[-1],[1],[0.5],[1.5]]) #c1 + 0.0
	parents = np.array([0, 1, 1, 1])
	g2 = fg.Forest(4,parents)
	
	dmax = 1

	i,j,k = BPmatch_slow_asym_dev(c1, c2, g1,g2, dmax)
	GM = np.zeros((5,5))
	GM[i,j]=k
	print GM*(GM>0.001)

	i,j,k = BPmatch_slow_asym_dev(c2, c1, g2,g1, dmax)
	GM = np.zeros((5,5))
	GM[i,j]=k
	print GM*(GM>0.001)

	i,j,k = BPmatch_slow_asym(c1, c2, g1,g2, dmax)
	GM = np.zeros((5,5))
	GM[i,j]=k
	print GM
	
	i,j,k = match_trivial(c1, c2, dmax, eps = 1.e-12 )
	GM = np.zeros((5,5))
	GM[i,j]=k
	print GM

###--------------------------------------------------------------------
###-------------- Deprected stuff -------------------------------------
###--------------------------------------------------------------------

def EDistance(X,Y):
	"""
	Computation of the euclidian distances between all the columns of X
	and those of Y
	"""
	if X.shape[0]!=Y.shape[0]:
		raise ValueError, "incompatible dimension for X and Y matrices"
	
	s1 = X.shape[1]
	s2 = Y.shape[1]
	NX = np.reshape(np.sum(X*X,0),(s1,1))
	NY = np.reshape(np.sum(Y*Y,0),(1,s2))
	ED = np.repeat(NX,s2,1)
	ED = ED + np.repeat(NY,s1,0)
	ED = ED-2*np.dot(np.transpose(X),Y)
	ED = np.maximum(ED,0)
	ED = np.sqrt(ED)
	return ED

def leaves(G):
	leaves = np.ones(G.V)
	leaves[G.edges[G.weights>0,1]]=0
	return leaves

def BPmatch_slow_asym_dep(c1, c2, G1, G2, dmax):
	"""
	New version which makes the differences between ascending
	and descending links
	"""
	if G1.V != c1.shape[0]:
		raise ValueError, "incompatible dimension for G1 and c1"

	if G2.V != c2.shape[0]:
		raise ValueError, "incompatible dimension for G2 and c2"

	eps = 1.e-7
	sqs = 2*dmax*dmax
	ofweight = np.exp(-0.5)

	# find the leaves of the graphs
	sl1 = (leaves(G1)).astype('bool')
	sl2 = (leaves(G2)).astype('bool')

	# get the distances
	D = Euclidian_distance(c1,c2)
	W = np.exp(-D*D/sqs);
	
	# cancel the weigts of non-leaves
	for i in np.where(sl1==0)[0]: W[i,:]=1	
	for i in np.where(sl1)[0]: W[i,sl2==0]=0

	# normalize the weights
	sW = np.sum(W,1)+ofweight
	W = np.transpose(np.transpose(W)/sW)
	W0 = W.copy()
	
	
	# get the different trees in each graph
	u1 = G1.cc()
	u2 = G2.cc()
	nt1 = u1.max()+1
	nt2 = u2.max()+1

	# get the tree-summed weights
	tW = np.zeros((G1.V,nt2))
	for i2 in range(nt2):
		tW[:,i2] = np.sum(W[:,u2==i2],1)
	
	# run the algo for each pair of tree
	for i1 in range(nt1):
		if np.sum(u1==i1)>1:
			g1 = G1.subgraph(u1==i1)
			for i2 in range(nt2):
				if np.sum(u2==i2)>1:		
					g2 = G2.subgraph(u2==i2)
					rW = W[u1==i1,:][:,u2==i2]
					rW = np.transpose(np.transpose(rW)/tW[u1==i1,i2])
					rW = _MP_algo_(g1,g2,rW,c1[u1==i1,:],c2[u2==i2],sqs)
					q = 0
					for j in np.where(u1==i1)[0]:
						W[j,u2==i2] = rW[q,:]*tW[j,i2]
						q = q+1
					
	# cancel the weigts of non-leaves
	W[sl1==0,:]=0
	W[:,sl2==0]=0

	W[W<1.e-4]=0
	i,j = np.where(W)
	k = W[i,j]
	return i,j,k

def _son_translation_dep(G,c):
	b = np.zeros((G.V,c.shape[1]))
	bb = np.zeros(G.V)
	for e in range(G.E):
		if G.weights[e]<0:
			ip = G.edges[e,0]
			target = G.edges[e,1]
			b[ip,:] = b[ip,:]+c[target,:]
			bb[ip] = bb[ip]+1
			
	for v in range(G.V):
		if bb[v]>0:
			b[v,:] = b[v,:]/bb[v]

	v = np.zeros((G.E,c.shape[1]))
	for e in range(G.E):
		if G.weights[e]<0:
			ip = G.edges[e,0]
			target = G.edges[e,1]
			v[e,:] = 2*(c[target,:]-b[ip,:])		

	return v

def _make_roots_(G,u):
	if G.E==0:
		return u 
	nbcc = u.max()+1
	edges = G.get_edges()
	edges = edges[G.weights>0,:]
	root = -1*np.ones(nbcc)
	for i in range(nbcc):
		candidates = (u==i)
		candidates[edges[:,0]] = 0
		r = np.nonzero(candidates)
		if np.size(r)>1:
			print np.nonzero(u==i),r,edges[:,1]
			raise ValueError, "Too many candidates"
		root[i] = np.reshape(r,np.size(r))
	return root
