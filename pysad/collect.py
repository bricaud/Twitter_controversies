
import pandas as pd
import random
from tqdm import tqdm





def process_hop(graph_handle, data_path, username_list):
	""" collect the tweets and tweet info of the users in the list username_list
	"""
	#print('Collecting the tweets for the last {} days.'.format(max_day_old))
	users_dic = {'username':[], 'Nb_diff_mentions': []}
	new_users_list = []
	empty_tweets_users = []
	total_edges_df = pd.DataFrame()
	total_nodes_df = pd.DataFrame()

	for user in tqdm(username_list):
		if not isinstance(user,str):
			continue
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
			# Save to json file
			#edgefilename = data_path + user + '_mentions' + '.json'
			#nodefilename = data_path + user + '_userinfo' + '.json'
			#edges_df.to_json(edgefilename)
			#node_df.to_json(nodefilename)
			# Extract mentioned users
			#edges_g = group_edges(edges_df)	
			#users_connected = edges_g['mention'][edges_g['weight']>=min_mentions]
			#new_users_list += users_connected.tolist()
		#else: # keep track of the users with empty account
		#	empty_tweets_users.append(user)
	#print('users with empty tweet list or no mention:',empty_tweets_users)
	#return new_users_list 


def collect_tweets(username_list, data_path, graph_handle, exploration_depth=4, random_sset=False):
	""" Collect the tweets of the users and their mentions
		make an edge list user -> mention
		and save each user edge list to a file
	"""
	print('Threshold set to {} mentions.'.format(graph_handle.rules['min_mentions']))
	print('Collecting the tweets for the last {} days.'.format(graph_handle.rules['max_day_old']))
	users_dic = {'username':[], 'Nb_mentions': [], 'mentions_of_mentions': []}
	total_username_list = []
	total_username_list += username_list
	new_username_list = username_list.copy()
	total_edges_df = pd.DataFrame()
	total_nodes_df = pd.DataFrame()
	for depth in range(exploration_depth):
		print('')
		print('******* Processing users at {}-hop distance *******'.format(depth))
		new_users_founds, edges_df, nodes_df = process_hop(graph_handle, data_path, new_username_list)
		#New users to collect:
		new_username_list = list(set(new_users_founds).difference(set(total_username_list))) # remove the one already collected
		
		
		if random_sset == True and len(new_username_list)>500:
			# Only explore a random subset of users
			random_subset_size = 200
			print('---')
			print('Too many users mentioned ({}). Keeping a random subset of {}.'.format(len(new_username_list),random_subset_size))
			new_username_list = random.sample(new_username_list, random_subset_size)		
		
		total_username_list += new_username_list
		total_edges_df = total_edges_df.append(edges_df)
		total_nodes_df = total_nodes_df.append(nodes_df)
	if len(total_username_list) < 100:
		print('Total number of users collected:')
		print(len(total_username_list),len(set(total_username_list)))	
		print('Low number of users, processing one more hop.')
		new_users_founds, edges_df, nodes_df = process_hop(graph_handle, data_path, new_username_list)
		#New users to collect:
		new_username_list = list(set(new_users_founds).difference(set(total_username_list))) # remove the one already collected
		total_username_list += new_username_list
		total_edges_df = total_edges_df.append(edges_df)
		total_nodes_df = total_nodes_df.append(nodes_df)

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