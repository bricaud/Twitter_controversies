
import pandas as pd
import random
from tqdm import tqdm





def process_hop(graph_handle, username_list):
	""" collect the tweets and tweet info of the users in the list username_list
	"""
	new_users_list = []
	#empty_tweets_users = []
	total_edges_df = pd.DataFrame()
	total_nodes_df = pd.DataFrame()

	for user in tqdm(username_list):
		edges_df, node_df = graph_handle.get_neighbors(user)
		# Collect mentioned users for the next hop
		if not edges_df.empty: # list of edges and their properties
			edges_df = graph_handle.filter_edges(edges_df)
			total_edges_df = total_edges_df.append(edges_df)
			neighbors = graph_handle.neighbors_list(edges_df)
			new_users_list += neighbors
		if not node_df.empty: # list of node properties
			node_df = graph_handle.filter_nodes(node_df)
			total_nodes_df = total_nodes_df.append(node_df)
	total_edges_df.reset_index(drop=True, inplace=True)
	total_nodes_df.reset_index(drop=True, inplace=True)

	return new_users_list, total_edges_df, total_nodes_df



def spiky_ball(username_list, graph_handle, exploration_depth=4, random_subset_size=None):
	""" Collect the tweets of the users and their mentions
		make an edge list user -> mention
		and save each user edge list to a file
	"""
	if graph_handle.rules:
		print('Parameters')
		for key,value in graph_handle.rules:
			print(key,value)
	total_username_list = []
	total_username_list += username_list
	new_username_list = username_list.copy()
	total_edges_df = pd.DataFrame()
	total_nodes_df = pd.DataFrame()
	for depth in range(exploration_depth):
		print('')
		print('******* Processing users at {}-hop distance *******'.format(depth))
		new_users_founds, edges_df, nodes_df = process_hop(graph_handle, new_username_list)
		#New users to collect:
		new_username_list = list(set(new_users_founds).difference(set(total_username_list))) # remove the one already collected
		
		if isinstance(random_subset_size,int) and (len(new_username_list)>random_subset_size):
			# Only explore a random subset of users
			print('---')
			print('Too many users mentioned ({}). Keeping a random subset of {}.'.format(len(new_username_list),random_subset_size))
			new_username_list = random.sample(new_username_list, random_subset_size)		
		
		total_username_list += new_username_list
		total_edges_df = total_edges_df.append(edges_df)
		total_nodes_df = total_nodes_df.append(nodes_df)
	
	# optional
	# if len(total_username_list) < 100:
	# 	print('Total number of users collected:')
	# 	print(len(total_username_list),len(set(total_username_list)))	
	# 	print('Low number of users, processing one more hop.')
	# 	new_users_founds, edges_df, nodes_df = process_hop(graph_handle, new_username_list)
	# 	#New users to collect:
	# 	new_username_list = list(set(new_users_founds).difference(set(total_username_list))) # remove the one already collected
	# 	total_username_list += new_username_list
	# 	total_edges_df = total_edges_df.append(edges_df)
	# 	total_nodes_df = total_nodes_df.append(nodes_df)

	total_edges_df.reset_index(drop=True, inplace=True)
	total_nodes_df.reset_index(drop=True, inplace=True)
	return total_username_list, total_nodes_df, total_edges_df

def save_data(nodes_df,edges_df,data_path):
	# Save to json file
	edgefilename = data_path + 'edges_data' + '.json'
	nodefilename = data_path + 'nodes_data' + '.json'
	print('Writing',edgefilename)
	edges_df.to_json(edgefilename)
	print('Writing',nodefilename)
	nodes_df.to_json(nodefilename)
	return None