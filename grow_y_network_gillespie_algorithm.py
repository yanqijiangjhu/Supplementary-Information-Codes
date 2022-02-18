import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 
import pdb

def initialize_system():
#using experimental data to initialize the system with the proper number of 1, 2, 3 armed structures
	system = nx.Graph()
	n_1arm_pos = 15
	n_2arm_pos = 40
	n_3arm_pos = 55

	n_1arm_neg = 15
	n_2arm_neg = 40
	n_3arm_neg = 55
	for i in range(n_1arm_pos):
		system.add_node(system.number_of_nodes()+1, arms = 1, adapter = 'pos')
	for i in range(n_2arm_pos):
		system.add_node(system.number_of_nodes()+1, arms = 2, adapter = 'pos')
	for i in range(n_3arm_pos):
		system.add_node(system.number_of_nodes()+1, arms = 3, adapter = 'pos')

	for i in range(n_1arm_neg):
		system.add_node(system.number_of_nodes()+1, arms = 1, adapter = 'neg')
	for i in range(n_2arm_neg):
		system.add_node(system.number_of_nodes()+1, arms = 2, adapter = 'neg')
	for i in range(n_3arm_neg):
		system.add_node(system.number_of_nodes()+1, arms = 3, adapter = 'neg')

	print "initial number of nodes: ", system.number_of_nodes()

	return system

def compute_rtot_first(system):
	
	graphs = list(nx.connected_component_subgraphs(system))
	print len(graphs)

	rtot = 0
	
	for i in range(len(graphs)):
		for j in range (i+1, len(graphs)):
			if i == j:
				continue
			else:
				network_1 = graphs[i]
				network_2 = graphs[j]
				kjoin = joining_rate(network_1, network_2)
				print kjoin 
				rtot += kjoin
	return rtot 

#Given a joining reaction, update rtot by subtracting all terms involving either of the joining partners and add terms involving joining to the new network:

def update_rtot(old_rtot, network_1, network_2, new_network, system_old)

	graphs = list(nx.connected_component_subgraphs(system_old))
	print len(graphs)

	rtot = old_rtot
	
	for i in range(len(graphs)):
		network = graphs[i]
		kjoin1 = joining_rate(network_1, network)
		rtot -= kjoin1
		print "subtracting: ", kjoin1
		
		kjoin2 = joining_rate(network_2, network)
		rtot -= kjoin2
		print "subtracting: ", kjoin2

		kjoin_new = joining_rate(new_network, network)
		rtot += kjoin_new 
		print "adding: ", kjoin_new 

	#To avoid double counting the interation of the two networks with themselves:		
	network_1_correction = joining_rate(network_1, network_1) 
	network_2_correction = joining_rate(network_2, network_2) 
	rtot += network_1_correction
	rtot += network_2_correction
	
	#To avoid counting the interaction of the new network with the two old networks:
	new_network_correction_1 = joining_rate(new_network, network_1) 
	new_network_correction_2 = joining_rate(new_network, network_2)
	rtot -= new_network_correction_1
	rtot -= new_network_correction_2


	print rtot 
	return rtot 

def joining_rate(network_1, network_2):
	 
	connectivity_dict1 = nx.get_node_attributes(network_1, 'arms')
	adapter_dict1 = nx.get_node_attributes(network_1, 'adapter')
	connectivity_dict2 = nx.get_node_attributes(network_2, 'arms')
	adapter_dict2 = nx.get_node_attributes(network_2, 'adapter')

	#count valid attachment points for network 1
	valid_positive_attachment_points_1 = 0
	valid_negative_attachment_points_1 = 0
	for node, narms in connectivity_dict1.iteritems():
		if network_1.degree(node) < narms and network_1.node[node]['adapter'] == 'pos':
			valid_positive_attachment_points_1 += narms - network_1.degree(node)
		if network_1.degree(node) < narms and network_1.node[node]['adapter'] == 'neg':
			valid_negative_attachment_points_1 += narms - network_1.degree(node)

	#count valid attachment points for network 2
	valid_positive_attachment_points_2 = 0
	valid_negative_attachment_points_2 = 0
	for node, narms in connectivity_dict2.iteritems():
		if network_2.degree(node) < narms and network_2.node[node]['adapter'] == 'pos':
			valid_positive_attachment_points_2 += narms - network_2.degree(node)
		if network_2.degree(node) < narms and network_2.node[node]['adapter'] == 'neg':
			valid_negative_attachment_points_2 += narms - network_2.degree(node)
			
	number_of_possible_connections = float(valid_positive_attachment_points_1*valid_negative_attachment_points_2 + valid_negative_attachment_points_1*valid_positive_attachment_points_2)
	
	kjoin = 1.0/(float(network_1.number_of_nodes()*float(network_2.number_of_nodes())))
	return kjoin*number_of_possible_connections

def compute_dt(rtot):
	#compute dt from the total reaction rate according to the exponential distribution
	dt = np.random.exponential(float(1.0/rtot))
	return dt

def choose_joining_reaction(rtot, system, joining_matrix, joining_matrix_index_dictionary):
	
	graphs = list(nx.connected_component_subgraphs(system))
	print "old number of connected graphs", len(graphs)

	#Use the joining matrix to determine the probabilities for all pairwise joining reactions
	pairwise_probabilities = joining_matrix.ravel()/np.sum(joining_matrix)
	ravel_indices = []
	for i in range(len(pairwise_probabilities)):
		ravel_indices.append(i)

	chosen_ravel_index = np.random.choice(ravel_indices, p = pairwise_probabilities)
	chosen_matrix_indices = np.unravel_index(chosen_ravel_index, joining_matrix.shape)

	print chosen_matrix_indices[0]
	index_of_network_1 = chosen_matrix_indices[0]
	index_of_network_2 = chosen_matrix_indices[1]
	lowest_node_in_network_1 = joining_matrix_index_dictionary[index_of_network_1]
	lowest_node_in_network_2 = joining_matrix_index_dictionary[index_of_network_2]

	for network in graphs:
		if network.has_node(lowest_node_in_network_1):
			network_1 = network
			break
	for network in graphs:
		if network.has_node(lowest_node_in_network_2):
			network_2 = network
			break

	print "lowest node in network 1:", lowest_node_in_network_1
	print "lowest node in network 2:", lowest_node_in_network_2
	
	
	return network_1, network_2, index_of_network_1, index_of_network_2

def perform_joining(network_1, network_2, system):
	
	connectivity_dict1 = nx.get_node_attributes(network_1, 'arms')
	connectivity_dict2 = nx.get_node_attributes(network_2, 'arms')
	valid_connection = False
	while valid_connection == False: 
		network_1_node = np.random.choice(network_1.nodes())
		network_2_node = np.random.choice(network_2.nodes())
		network_1_node_narms = connectivity_dict1[network_1_node]
		network_2_node_narms = connectivity_dict2[network_2_node]
		if network_1.degree(network_1_node) < network_1_node_narms and network_2.degree(network_2_node) < network_2_node_narms and network_1.node[network_1_node]['adapter'] != network_2.node[network_2_node]['adapter'] :
			print "adding valid connection"
			valid_connection = True
	system.add_edge(network_1_node, network_2_node)
	new_network = nx.union(network_1, network_2)
	new_network.add_edge(network_1_node, network_2_node)
	

	return system, new_network



def initialize_joining_matrix(system):
	joining_matrix = np.zeros((system.number_of_nodes(),system.number_of_nodes()))
	graphs = list(nx.connected_component_subgraphs(system))
	
	joining_matrix_index_dictionary = {}

	for i in range(len(graphs)):
		lowest_node_number = min(graphs[i].nodes())
		joining_matrix_index_dictionary[i] = lowest_node_number 
		for j in range (i+1, len(graphs)):
			network_1 = graphs[i]
			network_2 = graphs[j]
			kjoin = joining_rate(network_1, network_2)
			joining_matrix[i,j] = kjoin 
	return joining_matrix, joining_matrix_index_dictionary

def compute_rtot_from_joining_matrix(joining_matrix):
	
	matrix_sum = np.sum(joining_matrix)
	return matrix_sum

def update_joining_matrix(joining_matrix, joining_matrix_index_dictionary, joined_network_1, joined_network_2, new_network, index_1, index_2, system):
	
	lowest_node_1 = min(joined_network_1.nodes())
	lowest_node_2 = min(joined_network_2.nodes())

	joining_matrix = np.delete(joining_matrix, index_1, 0) #deletes the row for network 1
	joining_matrix = np.delete(joining_matrix, index_1, 1) #deletes the column for network 1
	updated_dict_1 = {}
	for key, value in joining_matrix_index_dictionary.iteritems():
		if key< index_1:
			#this means this entry is unaffected by the deletion
			updated_dict_1[key] = value 
		if key> index_1:
			#this means that the key should be reduced by 1 because of the deletion
			new_key = key - 1
			updated_dict_1[new_key] = value
		#this dictionary is now one key smaller than the origianl b/c key==index 1 was not covered
	print index_2
	
	if index_2>index_1:
		index_2-=1
	joining_matrix = np.delete(joining_matrix, index_2, 0) #deletes the row for network 2
	joining_matrix = np.delete(joining_matrix, index_2, 1) #deletes the column for network 2
 
	updated_dict_2 = {}
	for key, value in updated_dict_1.iteritems():
		if key< index_2:
			updated_dict_2[key] = value 
		if key> index_2:
			new_key = key - 1
			updated_dict_2[new_key] = value
	
	new_column = np.zeros(np.size(joining_matrix,1))
	joining_matrix = np.insert(joining_matrix,np.size(joining_matrix,1),new_column, axis=1)
	new_row = np.zeros(np.size(joining_matrix,1))
	joining_matrix = np.insert(joining_matrix,np.size(joining_matrix,0),new_row, axis=0)
	
	graphs = list(nx.connected_component_subgraphs(system))

	column = np.size(joining_matrix,1)-1
	new_network_lowest_node_number = min(new_network.nodes())
	for i in range(len(graphs)):
		network = graphs[i]

		lowest_node_number = min(network.nodes())
		if lowest_node_number == new_network_lowest_node_number:
			
			kjoin = 0
			#break
		else:
			kjoin = joining_rate(new_network, network)
			 
			for key, value in updated_dict_2.iteritems(): 
				if value == lowest_node_number:
					index = key
					break
					#pdb.set_trace()
			joining_matrix[index,column] = kjoin 

	
	#Update the dictionary
	updated_dict_2_original = updated_dict_2.copy()
	updated_dict_2[column]=new_network_lowest_node_number
	joining_matrix_index_dictionary = updated_dict_2
	#pdb.set_trace()
	return joining_matrix, joining_matrix_index_dictionary



def perform_gillespie_simulation(i):
	#np.random.seed(1337)
	#initialize the system according the experimental starting conditions
	system = initialize_system()

	number_of_joining_steps = system.number_of_nodes()-10 #we will stop just short of everything in the system being joined to itself
	
	average_nodes_per_network = [1]
	number_of_connected_graphs = [system.number_of_nodes()]
	average_seeds_per_network = [1]
	average_network_size_of_a_seed_list = [1]

	time = [0]

	#joining_matrix = np.zeros((system.number_of_nodes(),system.number_of_nodes()))
	joining_matrix, joining_matrix_index_dictionary = initialize_joining_matrix(system)
	print joining_matrix
	print joining_matrix_index_dictionary

	rtot = compute_rtot_from_joining_matrix(joining_matrix)

	print "rtot before loop starts: ", rtot 
	#joining_reactions_list, probabilities_list = generate_joining_reactions_list(system)
	for step in range(number_of_joining_steps):
		#print joining_matrix
		print "updated rtot: ", rtot
		#brute_force_rtot = compute_rtot_first(system)
		#print "brute force rtot: ", brute_force_rtot
		if rtot == 0.0:
			#no more reactions can occur
			break
		dt = compute_dt(rtot)
		previous_time = time[len(time)-1]
		time.append(previous_time + dt)


		joined_network_1, joined_network_2, index_1, index_2 = choose_joining_reaction(rtot, system, joining_matrix, joining_matrix_index_dictionary)

		system, new_network = perform_joining(joined_network_1, joined_network_2, system)

		new_joining_matrix, new_joining_matrix_index_dictionary = update_joining_matrix(joining_matrix, joining_matrix_index_dictionary, joined_network_1, joined_network_2, new_network, index_1, index_2, system)
		joining_matrix = new_joining_matrix
		joining_matrix_index_dictionary = new_joining_matrix_index_dictionary

		rtot_new = compute_rtot_from_joining_matrix(joining_matrix)
		rtot = rtot_new #updating rtot for the next iteration

		graphs = list(nx.connected_component_subgraphs(system))
		print "new number of connected graphs", len(graphs)
		number_of_connected_graphs.append(len(graphs))
		#average_seeds_per_network.append(50.0/float(len(graphs)))

		#Compute the average network size that a seed is in
		network_size_count = []
		for graph in graphs:
			for node in graph.nodes():
				network_size_count.append(graph.number_of_nodes())
		average_network_size_of_a_seed = float(sum(network_size_count))/float(len(network_size_count))
		average_network_size_of_a_seed_list.append(average_network_size_of_a_seed)

		#printing out data, do this every 200 time steps 
		if step%20 == 0:

				print "printing current simulation time"
				f1=open('simulation_time.dat', 'a')
				print >>f1, previous_time 
				f1.close()

				print "average network size of a seed"
				f1=open('average_network_size_of_a_seed.dat','a')
				print >>f1, average_network_size_of_a_seed
				f1.close()

				print "printing network sizes squared"
				f1=open(str(previous_time)+"_network_size_count_squared.dat", 'w+')
				for size in network_size_count:
					print >>f1, size
				f1.close()

				print "printing network sizes"
				f1=open(str(previous_time)+"_network_size_count.dat", 'w+')
				for graph in graphs:
					print >>f1, graph.number_of_nodes()
				f1.close()


	#print time
	#print number_of_connected_graphs
	#print average_seeds_per_network
	#print average_network_size_of_a_seed_list

