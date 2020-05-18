
import networkx as nx
import pandas as pd

class graph:


	rules = {}


	def __init__(self,graph):
		# Instantiate an object
		self.G = graph


	def get_neighbors(self,node_id):
		G = self.G
		if node_id not in G:
			return pd.DataFrame(),pd.DataFrame()
		# node data
		node_df = pd.DataFrame([{'source':node_id, **G.nodes[node_id]}])
		# Edges and edge data		
		if nx.is_directed(G):
			#inedges = G.in_edges(node_id, data=True)
			edges = G.out_edges(node_id, data=True)
		else:
			edges = G.edges(node_id, data=True)
		edgeprop_dic_list = []
		#nodeprop_dic_list = []
		for source,target,data in edges:
			edge_dic = {'source': source, 'target': target, **data}
			edgeprop_dic_list.append(edge_dic)
		edges_df = pd.DataFrame(edgeprop_dic_list)
		return edges_df,node_df

	def filter_nodes(self,nodes_df):
		return nodes_df

	def filter_edges(self,edges_df):
		#edges_g = self.group_edges(edges_df)	
		#users_to_remove = edges_g['mention'][edges_g['weight'] < self.rules['min_mentions']]
		# Get names of indexes for which column Age has value 30
		#indexNames = edges_df[ edges_df['user'].isin(users_to_remove) ].index
		# Delete these row indexes from dataFrame
		#edges_df.drop(indexNames , inplace=True)
		return edges_df

	def neighbors_list(self,edges_df):
		neighbors = edges_df['target'].unique().tolist()
		return neighbors



def reshape_node_data(nodes_df):
	nodes_df.set_index('source', inplace=True)
	return nodes_df

def reshape_edge_data(edge_df):
	edge_df.set_index(['source','target'], inplace=True)
	return edge_df