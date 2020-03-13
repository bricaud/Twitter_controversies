from datetime import date
import pysad.utils as pu
import pysad.graph as pg
import pysad.clusters as pcl
import glob
import pandas as pd
import networkx as nx
from tqdm import tqdm

## Options
# path
category_name = 'coronavirus'
#category_name = 'french_tech_lesechos'
today = date.today()
date_string = today.strftime("%Y%m%d")
tweet_data_path_list = ['..','tweetdata', category_name, date_string]
results_data_path_list = ['..','resultsdata2', category_name, date_string]
# Graph
DEGREE_MIN = 2 # Minimal number of connections in the graph
##
# Initialize folders (create or clean them if they exist)
tweet_data_path = pu.initialize_folder(tweet_data_path_list, erase=False)
results_data_path = pu.initialize_folder(results_data_path_list, erase=False)




edge_df = pd.DataFrame()
print('Reading path',tweet_data_path)
for filename in tqdm(glob.glob(tweet_data_path + '*_mentions*' + '.json')):
    new_edge_df = pd.read_json(filename)
    #print('{} with {} tweets.'.format(filename,len(new_edge_df)))
    edge_df = edge_df.append(new_edge_df)
edge_df.reset_index(drop=True, inplace=True)

# Graph

G = pg.graph_from_edgeslist(edge_df,DEGREE_MIN)
G.name = category_name
G.end_date = max(edge_df['date']) #max(edge_df['date'].apply(max))
G.start_date = min(edge_df['date']) #min(edge_df['date'].apply(min))
print('Period from {} to {}.'.format(G.start_date,G.end_date))

G,clusters = pg.detect_communities(G)
G.nb_communities = len(clusters)
c_connectivity = pg.cluster_connectivity(G)

# Save the graph

graphname = 'globalgraph'
graphfilename = results_data_path + graphname + '_md' + str(DEGREE_MIN) +'_graph.gexf'
nx.write_gexf(G,graphfilename)
print('Wrote',graphfilename)

# Extracting the data from the clusters
print('Extracting data from clusters...')
cluster_info_dic = {}
for c_id in tqdm(clusters):
    cgraph = clusters[c_id]
    cgraph = pcl.cluster_attributes(cgraph)
    table_dic = pcl.cluster_tables(cgraph)
    cluster_filename = results_data_path + 'cluster' + str(c_id)
    cluster_info_dic[c_id] = {}
    cluster_info_dic[c_id]['info_table'] = table_dic
    cluster_info_dic[c_id]['filename'] = cluster_filename    

# Adding global infos
# keywords
corpus = pcl.get_corpus(cluster_info_dic)
keyword_dic = pcl.tfidf(corpus)

# gathering global info
# Saving in excel files
for c_id in cluster_info_dic:
    info_table = cluster_info_dic[c_id]['info_table']
    info_table['keywords'] = keyword_dic[c_id]
    cluster_general_info = {'cluster id': c_id, 'Nb users': clusters[c_id].number_of_nodes(), 
                           'Nb of tweets':clusters[c_id].size(weight='weight'),
                           'Start date': str(G.start_date),
                           'End date': str(G.end_date),
                           'Search topic': category_name,
                           'cluster connectivity': c_connectivity[c_id]}
    cluster_general_df = pd.DataFrame.from_dict([cluster_general_info])
    #info_table = {'cluster':cluster_general_df, **info_table}
    sheet1 = pd.concat([cluster_general_df,info_table['hashtags'],info_table['keywords']],axis=1)
    tweet_table = info_table['text']
    cluster_indicators = pd.DataFrame([pcl.compute_cluster_indicators(clusters[c_id])])
    excel_data = {'cluster':sheet1, 'tweets':tweet_table, 'indicators': cluster_indicators}
    #excel_data = info_table
    pcl.save_excel(excel_data,cluster_info_dic[c_id]['filename'] + '_infos.xlsx', table_format='Fanny')
    pg.save_graph(clusters[c_id],cluster_info_dic[c_id]['filename'] + 'graph.gexf')

