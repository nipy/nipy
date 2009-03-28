import numpy as	 np
import fff2.graph.graph as fg

def mesh_to_graph(mesh):
	"""
	This function builds an fff graph from a mesh
	"""
	vertices = np.array(mesh.vertex())
	poly  = mesh.polygon()

	V = len(vertices)
	E = poly.size()
	edges = np.zeros((3*E,2))
	weights = np.zeros(3*E)
	poly  = mesh.polygon()

	for i in range(E):
		sa = poly[i][0]
		sb = poly[i][1]
		sc = poly[i][2]
		
		edges[3*i] = np.array([sa,sb])
		edges[3*i+1] = np.array([sa,sc])
		edges[3*i+2] = np.array([sb,sc])	
			
	G = fg.WeightedGraph(V,edges,weights)

	# symmeterize the graph
	G.symmeterize()

	# remove redundant edges
	G.cut_redundancies()

	# make it a metric graph
	G.set_euclidian(vertices)

	return G

def flatten(mesh):
	"""
	This function flattens the input mesh
	"""
	import fff.NLDR
	G = mesh_to_graph(mesh)

	chart = fff.NLDR.isomap_dev(G,dim=2,p=300,verbose = 0)
	
	#print np.shape(chart)
	vertices = np.array(mesh.vertex())
	
	for i in range(G.V):
		mesh.vertex()[i][0]=chart[i,0]
		mesh.vertex()[i][1]=chart[i,1]
		mesh.vertex()[i][2]= 0
		
	mesh.updateNormals()
	
	return mesh

def write_aims_Mesh(vertex, polygon, fileName):
	"""
	Given a set of vertices, polygons and a filename,
	write the corresponding aims mesh
	the aims mesh is returned
	"""
	from soma import aims
	vv = aims.vector_POINT3DF()
	vp = aims.vector_AimsVector_U32_3()
	for x in vertex: vv.append(x)
	for x in polygon: vp.append(x)
	m = aims.AimsTimeSurface_3()
	m.vertex().assign( vv )
	m.polygon().assign( vp )
	m.updateNormals()
	W = aims.Writer()
	W.write(m, fileName)
	return m
