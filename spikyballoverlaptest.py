import pandas as pd
#import json
from datetime import date
import pysad
import pysad.utils
import pysad.collect
import pysad.twitter
import pysad.graph
import json

credential_file = "twitter_credentials.json"
graph_handle = pysad.twitter.twitter_network(credential_file)

init_accounts = pysad.utils.initial_accounts()

######Choose a category##############    
#category_name = 'swiss_climate_controversial'
#category_name = 'swiss_climate_regular'
#category_name = 'french_tech_lesechos'
#category_name = 'swiss_immigration'
#category_name = 'swiss_immigration2'
#category_name = 'debat_burqa'
category_name = 'hackathlon'
category_name = 'hackathlon_popular'
category_name = 'hackathlon_reddit'
category_name = 'crevecoeur'
category_name = 'voat'
category_name = 'fakemedicine'
category_name = 'Benjamin'
#category_name = 'Nicolas'
#category_name = 'Vlad'
#category_name = 'hackathlon_missingtweets'

#####################################

username_list = init_accounts.accounts(category_name)

# create the path to save the experiment indexed with the date of today
today = date.today()
date_string = today.strftime("%Y%m%d")
print("date string =", date_string)

tweet_data_path_list = ['../tweetdata', category_name, date_string]
results_data_path_list = ['../resultsdata2', category_name, date_string]

tweet_data_path = ''.join(tweet_data_path_list)
results_data_path = ''.join(results_data_path_list)
# Initialize folders (create or clean them if they exist)
# Set erase=False if you need to keep the previous collection
tweet_data_path = pysad.utils.initialize_folder(tweet_data_path_list, erase=True)
results_data_path = pysad.utils.initialize_folder(results_data_path_list, erase=False)

##### parameters
graph_handle.rules['min_mentions'] = 2 # minimal number of mentions of a user to be followed
graph_handle.rules['max_day_old'] = 90 # number max of days in the past
exploration_depth = 10 # mention of mention of mention of ... up to exploration depth

test_dic = {}
#### Main loop, data collection
nb_loops = 7
for i in range(nb_loops):
	print('#####################')
	print('## Processing step {}/{}'.format(i,nb_loops))
	print('#####################')
	total_user_list, total_nodes_df, total_edges_df = pysad.collect.collect_tweets(username_list, 
																				   graph_handle, 
																				   exploration_depth=exploration_depth,
																				   random_subset_size=200)
	node_df = total_nodes_df
	edge_df = total_edges_df
	node_df = pysad.graph.reshape_node_data(node_df)

	#### Creating the graph
	MIN_WEIGHT = 2
	MIN_DEGREE = 2 # Minimal number of connections in the graph

	G = pysad.graph.graph_from_edgeslist(edge_df, MIN_WEIGHT)
	#G = pysad.graph.graph_from_edgeslist(df_pop,DEGREE_MIN)
	G = pysad.graph.add_node_attributes(G,node_df)
	G = pysad.graph.reduce_graph(G,MIN_DEGREE)
	# Warning, graph properties are not saved by networkx in gexf files except graph name
	#G.graph['end_date'] = max(edge_df['date']).strftime("%d/%m/%Y") 
	#G.graph['start_date'] = min(edge_df['date']).strftime("%d/%m/%Y")
	#G.graph['name'] = category_name + ' ' + G.graph['start_date'] + ' - ' + G.graph['end_date'] 
	#print('Period from {} to {}.'.format(G.graph['start_date'],G.graph['end_date']))

	# Complete the info of the nodes not collected
	nodes_missing_info = [node for node,data in G.nodes(data=True) if 'name' not in data]
	print('Number of nodes with missing info:',len(nodes_missing_info))

	# remove nodes with missing info
	print('Removing node with missing info from the graph')
	G.remove_nodes_from(nodes_missing_info)
	print('Number of nodes after removal:',G.number_of_nodes())
	nodedegree_dic = dict(G.degree())
	test_dic[i] = nodedegree_dic


# Serialize data into file:
savefile = "testoverlap3.json"
json.dump( test_dic, open( savefile, 'w' ) )
print('wrote', savefile)
