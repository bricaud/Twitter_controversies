# Module for #sad project

import pandas as pd
import os
import json
from datetime import datetime, timedelta, date
from twython import TwythonError
import preprocessor as tweetpre
from collections import Counter
import numpy as np
import networkx as nx
import community

from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm


###############################################################
# Functions for extracting tweet info from the twitter API
###############################################################

def fill_retweet_info(tweet_dic,raw_retweet):
	# handle the particular structure of a retweet to get the full text retweeted
	tweet_dic['retweeted_from'].append(raw_retweet['user']['screen_name'])
	if raw_retweet['truncated']:
		full_text = raw_retweet['extended_tweet']['full_text']
	else:
		full_text = raw_retweet['full_text']
	return tweet_dic, full_text

def get_full_url(url_dic):
	if 'unwound' in url_dic:
		return url_dic['unwound']['url']
	return url_dic['expanded_url']

def get_user_tweets(tweet_handle, username,count=100, max_day_old=None):
	# Collect tweets from a username
	tweets_dic = {'user': [], 'date': [], 'text': [], 'favorite_count': [], 'retweet_count': [],
		'user_mentions': [], 'urls': [], 'hashtags': [], 'place': [], 'retweeted_from': []}

	# Test if ok
	try:
		lasttweet = tweet_handle.get_user_timeline(screen_name = username,  
										   count = 1, include_rts = True, tweet_mode='extended')
	except TwythonError as e:
		print('Twitter API returned error {} for user {}.'.format(e.error_code, username))
		return tweets_dic
	for raw_tweet in tweet_handle.get_user_timeline(screen_name = username,  
										   count = count, include_rts = True, tweet_mode='extended'):
		# Meta data
		time_struct = datetime.strptime(raw_tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
		ts = time_struct.strftime('%Y-%m-%d %H:%M:%S')
		if (max_day_old is not None) and (time_struct < datetime.now() - timedelta(days = max_day_old)):
			break # stop iterating on the tweet list
		tweets_dic['user'].append(raw_tweet['user']['screen_name'])
		tweets_dic['date'].append(ts)
		tweets_dic['favorite_count'].append(raw_tweet['favorite_count'])
		tweets_dic['retweet_count'].append(raw_tweet['retweet_count'])    
		tweets_dic['user_mentions'].append([user['screen_name'] for user in raw_tweet['entities']['user_mentions']])
		tweets_dic['urls'].append([get_full_url(url) for url in raw_tweet['entities']['urls']])
		tweets_dic['hashtags'].append([htg['text'] for htg in raw_tweet['entities']['hashtags']])
		#if raw_tweet['entities']['hashtags']:
		#    print([htg['text'] for htg in raw_tweet['entities']['hashtags']])
		#print(raw_tweet)
		if 'place' in raw_tweet and raw_tweet['place'] != None:          
			tweets_dic['place'].append(raw_tweet['place']['name'])
		else:
			tweets_dic['place'].append(None)
		
		# Handle text and retweet data
		if raw_tweet['truncated']:
			full_text = raw_tweet['extended_tweet']['full_text']
		else:
			full_text = raw_tweet['full_text']    
		if 'retweeted_status' in raw_tweet:
			tweets_dic, full_text = fill_retweet_info(tweets_dic,raw_tweet['retweeted_status'])
		else:
			tweets_dic['retweeted_from'].append(None)
		tweets_dic['text'].append(full_text)
	return tweets_dic

##################################################################################
# Functions for turning tweet data into an edge list with properties
##################################################################################

def get_mentions_edges(tweet_df):
	# Create the user -> mention table with their properties fom the list of tweets of a user
	
	# Some bots to be removed from the collection
	usertoremove_list = ['threader_app','threadreaderapp']

	row_list = []
	for idx,tweet in tweet_df.iterrows():
		user = tweet['user']
		mentions = tweet['user_mentions']
		hashtags = tweet['hashtags']
		tweet_date = tweet['date']
		urls = tweet['urls']
		text = tweet['text']
		retweet_count = tweet['retweet_count']
		favorite_count = tweet['favorite_count'] 
		for m in mentions:
			if m == user: # skip self-mentions
				continue
			if m in usertoremove_list:
				continue
			row_list.append({'user':user,'mention': m, 'weight': 1, 'hashtags': hashtags,
							'date': tweet_date, 'urls':urls, 'text':text,
							'retweet_count': retweet_count,'favorite_count': favorite_count})
	mention_df = pd.DataFrame(row_list)
	return mention_df

def collect_user_mention(username,python_tweets,data_path, max_day_old):
	# Return the mentions of a users from its tweets, together with the hashtags of the tweet where the mention is
	tweets_dic = get_user_tweets(python_tweets,username,count=100, max_day_old=max_day_old)
	if not tweets_dic:
		print('User {} has an empty tweet list.'.format(username))
		return pd.DataFrame()
	#print(tweets_dic)
	tweet_df = pd.DataFrame(tweets_dic)
	mention_df = get_mentions_edges(tweet_df)
	return mention_df

def create_user_edgelist_new(python_tweets, data_path, username, thres, max_day_old):
	# Process the user username and its mentioned users
	# save in a file the edgelist for the user and each mentioned user
	mention_df = collect_user_mention(username,python_tweets,data_path, max_day_old=max_day_old)
	if mention_df.empty:
		return mention_df
	mentionfilename = data_path + username + '_mentions' +'_t' +str(thres)+'.json'
	mention_df.to_json(mentionfilename)
	return mention_df

# def group_edges(edge_df):
# 	# this agg only works with pandas version >= 0.25
# 	mention_grouped = edge_df.groupby(['user','mention']).agg(weight=('weight',sum),
# 																 hashtags=('hashtags', sum),
# 																 date=('date', sum),
# 																 urls=('urls', sum),
# 																 text=('text', sum))
# 																 #,date=('date',lambda x: mean(x)))#lambda x: list(x)))
# 	mention_grouped.reset_index(level=['user', 'mention'], inplace=True)
# 	return mention_grouped

def group_edges(edge_df):
	mention_grouped = edge_df.groupby(['user','mention']).agg(weight=('weight',sum))
	mention_grouped.reset_index(level=['user', 'mention'], inplace=True)
	return mention_grouped


def process_user_list(python_tweets, data_path, username_list, thres=3, max_day_old=None):
	""" collect the tweets and tweet info of the users in the list username_list
	"""
	users_dic = {'username':[], 'Nb_diff_mentions': []}
	print('Collecting the tweets for the last {} days.'.format(max_day_old))
	new_users_list = []
	empty_tweets_users = []
	for user in tqdm(username_list):
		mentions_df = create_user_edgelist_new(python_tweets, data_path, user, thres=thres, max_day_old=max_day_old)
		# Collect mentioned users for the next hop
		# Only collect the ones mentioned more than the threshold thres 
		if not mentions_df.empty:
			mentions = group_edges(mentions_df)	
			users_mentioned = mentions['mention'][mentions['weight']>thres]
			new_users_list += users_mentioned.tolist()
		else:
			empty_tweets_users.append(user)
	print('users with empty tweet list or no mention:',empty_tweets_users)
	return new_users_list 

def process_hop(depth, python_tweets, data_path, username_list, min_mentions, max_day_old):
	print('')
	print('******* Processing users at {}-hop distance *******'.format(depth))
	new_users_list = process_user_list(python_tweets, data_path, username_list, 
													thres=min_mentions, max_day_old=max_day_old)
	return new_users_list

def collect_tweets(username_list, data_path, python_tweets, min_mentions=2, max_day_old=7, exploration_depth=4):
	""" Collect the tweets of the users and their mentions
		make an edge list user -> mention
		and save each user edge list to a file
	"""
	print('Threshold set to {} mentions.'.format(min_mentions))
	print('Collecting the tweets for the last {} days.'.format(max_day_old))
	users_dic = {'username':[], 'Nb_mentions': [], 'mentions_of_mentions': []}
	total_username_list = []
	total_username_list += username_list
	new_username_list = username_list.copy()
	for depth in range(exploration_depth):
		new_users_founds = process_hop(depth, python_tweets, data_path, new_username_list, 
			min_mentions, max_day_old)
		#New users to collect:
		new_username_list = list(set(new_users_founds).difference(set(total_username_list))) # remove the one already collected
		total_username_list += new_username_list
	
	if len(total_username_list) < 100:
		print('Total number of users collected:')
		print(len(total_username_list),len(set(total_username_list)))	
		print('Low number of users, processing one more hop.')
		new_users_founds = process_hop(depth+1, python_tweets, data_path, new_username_list, 
			min_mentions, max_day_old)
		#New users to collect:
		new_username_list = list(set(new_users_founds).difference(set(total_username_list))) # remove the one already collected
		total_username_list += new_username_list
	print('Total number of users collected:')
	print(len(total_username_list),len(set(total_username_list)))
	return total_username_list

#############################################################
# Functions for the graph of users
#############################################################

def converttojson(edge_df):
	""" Check if column type is list or dict and convert it to json
		list or dict can not be saved using gexf or graphml format.
	"""
	edge_df_str = edge_df.copy()
	for idx,col in enumerate(edge_df.columns):
		first_row_element = edge_df.iloc[0,idx]
		if isinstance(first_row_element,list) or isinstance(first_row_element,dict):
			edge_df_str[col] = edge_df[col].apply(json.dumps)
			print('Field "{}" of class {} converted to json string'.format(col,type(first_row_element)))
		#else:
		#	print(col,type(edge_df[col][0]))
	return edge_df_str

#def aggregate_edges(edge_df):
	#TODO

def reshape_to_graph_edge_list(edge_grouped):
	edge_list = []
	for name,group in edge_grouped:
		json_df = group.to_json()
		edge_dic = {'user': name[0], 'mention': name[1], 'weight':group['weight'].sum(),
					'tweets': json_df}
		edge_list.append(edge_dic)
	return pd.DataFrame(edge_list)

def graph_from_edgeslist(edge_df,degree_min):
	print('Creating the graph from the edge list')
	edge_grouped = edge_df.groupby(['user','mention'])
	#edge_df_str = converttojson(edge_df)
	edge_df_str = reshape_to_graph_edge_list(edge_grouped)
	G = nx.from_pandas_edgelist(edge_df_str,source='user',target='mention', 
		edge_attr=['weight','tweets'], create_using=nx.DiGraph)
	print('Nb of nodes:',G.number_of_nodes())
	# Drop node with small degree
	remove = [node for node,degree in dict(G.degree()).items() if degree < degree_min]
	G.remove_nodes_from(remove)
	print('Nb of nodes after removing nodes with degree strictly smaller than {}: {}'.format(degree_min,G.number_of_nodes()))
	isolates = list(nx.isolates(G))
	G.remove_nodes_from(isolates)
	print('removed {} isolated nodes.'.format(len(isolates)))
	if G.is_directed():
		print('Warning: the graph is directed.')
	return G

def detect_communities(G):
	#first compute the best partition
	if isinstance(G,nx.DiGraph):
		Gu = G.to_undirected()
	else:
		Gu = G
	partition = community.best_partition(Gu, weight='weight')
	nx.set_node_attributes(G,partition,name='community')
	print('Communities saved on the graph as node attributes.')
	nb_partitions = max(partition.values())+1
	print('Nb of partitions:',nb_partitions)
	# Create a dictionary of subgraphs, one per community
	community_dic = {}
	for idx in range(nb_partitions):
		subgraph = G.subgraph([key for (key,value) in partition.items() if value==idx])
		community_dic[idx] = subgraph
	#clusters_modularity = community.modularity(partition, Gu)
	return G, community_dic #,clusters_modularity

#############################################################
## Functions for cluster analysis
#############################################################

def cluster_connectivity(G, weight='weight'):
	""" Compute the ratio nb of edges inside the community over nb edges pointing outside,
		for each community
	"""
	# 1) indexing the edges by community
	sum_edges_dic = { com : {} for com in range(G.nb_communities)}
	for node1,node2 in G.edges():
		comm1 = G.nodes[node1]['community']
		comm2 = G.nodes[node2]['community']
		if comm2 not in sum_edges_dic[comm1]:
			sum_edges_dic[comm1][comm2] = 0
			sum_edges_dic[comm2][comm1] = 0
		else:
			if weight == None:
				sum_edges_dic[comm1][comm2] += 1
				sum_edges_dic[comm2][comm1] += 1
			else:	
				sum_edges_dic[comm1][comm2] += G.edges[node1,node2][weight]
				sum_edges_dic[comm2][comm1] += G.edges[node1,node2][weight]
	c_connectivity = {}
	# 2) computing the connectivity
	for com in sum_edges_dic:
		in_out_edges = sum(sum_edges_dic[com].values())
		c_connectivity[com] = round(- np.log2(sum_edges_dic[com][com] / in_out_edges),3)   
	return c_connectivity

def cluster_attributes(cluster_graph):
	""" Compute and store structure properties in the cluster (to nodes and edges)
	"""
	cg = cluster_graph
	nx.set_node_attributes(cg,dict(nx.degree(cg)),'degree')
	if cg.is_directed():
		nx.set_node_attributes(cg,dict(cg.in_degree()),'in degree')
		nx.set_node_attributes(cg,dict(cg.out_degree()),'out degree')
	nx.set_node_attributes(cg,dict(nx.degree(cg,weight='weight')),'degree_w')
	nx.set_node_attributes(cg,nx.betweenness_centrality(cg),'bcentrality')
	nx.set_node_attributes(cg,nx.pagerank(cg),'pagerank')
	if cg.is_directed():
		nx.set_node_attributes(cg,nx.pagerank(cg.reverse()),'pagerank inv')
	nx.set_edge_attributes(cg,nx.edge_betweenness_centrality(cg),'bcentrality')
	return cg

def compute_cluster_indicators(subgraph):
	gc = subgraph
	gc_size = gc.number_of_nodes()
	ck = nx.algorithms.core.core_number(gc)
	max_k = max(ck.values())
	kcurve = [len([key for (key,value) in ck.items() if value==idx]) for idx in range(max_k+1)]
	max_k_core_size = kcurve[-1]
	density = gc.number_of_edges()/(gc_size*(gc_size-1))
	mean_edge_weight = gc.size(weight='weight')/gc.number_of_edges()
	sink_list = [node for node, out_degree in gc.out_degree() if out_degree == 0]
	source_list = [node for node, in_degree in gc.in_degree() if in_degree == 0]
	info_dic = {'nb_nodes': gc_size, 'k_max':max_k, 'max_kcore':max_k_core_size,
				'norm_kcore':max_k_core_size/gc_size,'density': density,
				'activity_per_edge': mean_edge_weight, 'nb_influencers': len(sink_list),
				'nb anthousiasts': len(source_list),
				'hierarchy':max_k*1/(max_k_core_size/gc_size), 'hierarchy2': max_k**2/gc_size}
	return info_dic

def indicator_table(cluster_dic):
	comm_list = []
	for c in cluster_dic:
		gc = cluster_dic[c]
		comm_dic = {'Community': c}
		info_dic = compute_cluster_indicators(gc)
		comm_dic = {**comm_dic,**info_dic}
		comm_list.append(comm_dic)
	community_table = pd.DataFrame(comm_list)
	return community_table

def cluster_textandinfo(subgraph):
	user_text = {}
	hashtags = []
	date_list = []
	urls = []
	for node1,node2,data in subgraph.edges(data=True):
		if node1 == node2:
			print('Self edge',node1)
		hashtags += json.loads(data['hashtags'])
		date_list += json.loads(data['date'])
		urls += json.loads(data['urls'])
		texts = json.loads(data['text'])
		# associate text to writer
		if node1 not in user_text:
			user_text[node1] = texts
		else:
			user_text[node1] += texts
	return user_text, hashtags, date_list, urls


def community_tags_dic(tags_list,nb_tags=None):
	# Create a dict with popular hashtags for each community
	# from collections import Counter
	htag_dic = {}
	most_common = Counter(tags_list).most_common(nb_tags)
	if nb_tags is None: # take all the hashtags
		nb_tags = len(most_common)
	for htag_idx in range(nb_tags): # filling the table with the hashtags
		if htag_idx < len(most_common): 
			htag_dic['hashtag'+str(htag_idx)] = most_common[htag_idx][0]
		else: # less hashtags than required
			htag_dic['hashtag'+str(htag_idx)] = ''
	return htag_dic

def hashtag_count_table(tags_list):
	# Create a table with hashtags and their count
	# from collections import Counter
	htag_list = []
	most_common = Counter(tags_list).most_common()
	for htag_idx in range(len(most_common)): # creting a list of dic with the hashtags
		htag_dic = {'hashtag': most_common[htag_idx][0], 'count': most_common[htag_idx][1]}
		htag_list.append(htag_dic)
	htag_table = pd.DataFrame(htag_list)
	return htag_table

def community_date_stats(dates_list):
	# Create a dict with mean time and deviation
	meandate,stddate = compute_meantime(dates_list)
	date_dic = {'Average date':meandate.date(), 'Deviation (days)':stddate.days}
	return date_dic


def dates_tags_table(cluster_dic):
	comm_list = []
	for c in cluster_dic:
		gc = cluster_dic[c]
		comm_dic = {'Community': c}
		user_text, hashtags, date_list, urls = cluster_textandinfo(gc)
		hash_dic = community_tags_dic(hashtags,nb_tags=5)
		date_dic = community_date_stats(date_list)
		comm_dic = {**comm_dic, **date_dic, **hash_dic, 'urls': urls, 'text': user_text}
		comm_list.append(comm_dic)
	community_table = pd.DataFrame(comm_list)
	return community_table		



def count_order_items(item_list,item_name):
	dic_list = []
	most_commons = Counter(item_list).most_common()
	for item_idx in range(len(most_commons)): # creating a list of dic with the hashtags
		item_dic = {item_name: most_commons[item_idx][0], 'count': most_commons[item_idx][1]}
		dic_list.append(item_dic)
	item_table = pd.DataFrame(dic_list)
	return item_table

def tokenize(text):
	#from nltk import word_tokenize
	#from nltk.corpus import stopwords
	stop_words = stopwords.words('french') + list(punctuation)
	words = word_tokenize(text)
	words = [w.lower() for w in words]
	return [w for w in words if w not in stop_words and not w.isdigit()]

def most_common_words(text_table):
	""" Requires nltk
	"""
	fulltext = ''
	for text in text_table:
		fulltext += ' ' + text
	
	tktext = tokenize(fulltext)
	word_table = count_order_items(tktext,'word')
	# Calculate frequency distribution
	#fdist = nltk.FreqDist(tktext)
	#return fdist.most_common()
	return word_table

def extract_info_from_cluster_table(cluster_edge_table):
	text_list = []
	htag_list = []
	url_list = []
	for index,row in cluster_edge_table.iterrows():
		username = row['user']
		tweet_df = pd.read_json(row['tweets'])
		
		for idx,tweet in tweet_df.iterrows():
			htags = tweet['hashtags']
			urls = tweet['urls']
			text = tweet['text']
			retweet_count = tweet['retweet_count']
			favorite_count = tweet['favorite_count']

			parsed_tweet = tweetpre.parse(text)
			# extract emojis
			emojis = []
			if parsed_tweet.emojis is not None:
				emojis = [emo.match for emo in parsed_tweet.emojis]
			tweetpre.set_options(tweetpre.OPT.MENTION, tweetpre.OPT.URL)
			filtered_text = tweetpre.clean(text)
			tweetpre.set_options()
			#emojis = parsed_tweet.emojis.match ???
			url_c = [convert_bitly_url(url_string) for url_string in urls]
			text_list.append({'text': text, 'user': username, 'url': url_c , 'emojis':emojis , 
				'retweet_count':retweet_count, 'favorite_count': favorite_count, 'filtered text': filtered_text, 
				'bcentrality': row['bcentrality']})
			htag_list += htags
			url_list += urls
	text_df = pd.DataFrame(text_list)
	mostcommon_words_df = most_common_words(text_df['filtered text'])
	hashtags_df = count_order_items(htag_list,'hashtag')
	url_df = count_order_items(url_list,'url')
	url_df = convert_bitly_table(url_df)
	filtered_url_df = drop_twitter_urls(url_df)
	return {'text': text_df, 'hashtags': hashtags_df, 'words': mostcommon_words_df, 'urls': filtered_url_df}


def cluster_tables(cluster_graph):
	edge_data_list = []
	cluster_users_df = pd.DataFrame.from_dict(dict(cluster_graph.nodes(data=True)),orient='index').sort_values('pagerank',ascending=False)
	cluster_users_df = cluster_users_df.drop('community',axis=1)
	cluster_users_df = cluster_users_df.reset_index().rename(columns={'index':'username'})
	for node1,node2,data in cluster_graph.edges(data=True):
		# Add writer
		data['user'] = node1
		edge_data_list.append(data)
	cluster_edge_info = pd.DataFrame(edge_data_list)
	cluster_edge_info = cluster_edge_info.sort_values('bcentrality',ascending=False)
	table_dic = extract_info_from_cluster_table(cluster_edge_info)
	return {'users': cluster_users_df, **table_dic}



def save_excel(table_dic,filename, table_format='full'):
	#import pandas.io.formats.excel
	#pandas.io.formats.excel.header_style = None

	with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:

		if table_format is 'full':
			for tablename in table_dic:
				col_size_dic = {'B:E':25, 'A:A':100}
				write_excel_sheet(table_dic[tablename],tablename,writer,col_size_dic)
		else:
			#table_dic = excel_reshape(table_dic)
			tablename = 'cluster'
			col_size_dic = {'A:B':10, 'C:C': 15,'D:E':20, 'F:F':20, 'G:G': 20, 'H:H':25 ,'J:J': 25}
			write_excel_sheet(table_dic[tablename],tablename,writer,col_size_dic)
			tablename = 'tweets'
			col_size_dic = {'A:A':100, 'B:B':20, 'C:D':50, 'E:E':25}
			if 'filtered text' in table_dic[tablename].columns:
				table_dic[tablename].drop(labels='filtered text',axis=1, inplace=True)
			write_excel_sheet(table_dic[tablename],tablename,writer,col_size_dic)
			tablename = 'indicators'
			col_size_dic = {'A:B':10, 'C:C': 15,'D:I':20}
			write_excel_sheet(table_dic[tablename],tablename,writer,col_size_dic)
		#for tablename in table_dic:
		#	write_excel_sheet(table_dic[tablename],tablename,writer,col_size_dic)
			#table_dic[tablename].to_excel(writer, sheet_name=tablename, index=False)
			#worksheet = writer.sheets[tablename]
			#worksheet.set_column('B:E',column_width,format1)
			#worksheet.set_column('A:A',100,format1)
	print('Data saved to',filename)		
	 

def write_excel_sheet(df,sheet_name,writer,column_size_dic):
	df.to_excel(writer, sheet_name=sheet_name, index=False)
	workbook  = writer.book
	# Add a header format.
	format1 = workbook.add_format({
			#'bold': True,
			'text_wrap': True,
			#'valign': 'top',
			#'fg_color': '#D7E4BC',
			'border': 1}) 
	# column sizes
	worksheet = writer.sheets[sheet_name]
	for column in column_size_dic:
		column_width = column_size_dic[column]
		worksheet.set_column(column,column_width,format1)

def excel_reshape(table_dic):
	# Sort table_dic
	hashtags_keywords = pd.concat([table_dic['cluster'],
		table_dic['hashtags'],table_dic['keywords']],axis=1)
	tweets = table_dic['text']
	table_dic_s = {'cluster': table_dic['cluster'],'hashtags': hashtags_keywords,
					'text': tweets}
	return table_dic_s

def save_graph(graph,graphfilename):
	nx.write_gexf(graph,graphfilename)
	print('Graph saved to',graphfilename)


#############################################################
## Global graph processing from clusters info
#############################################################

def get_corpus(clusters_dic):
	""" Return a corpus with one document per cluter.
		Each document contains all the cluster tweets concatenated.
	"""
	corpus = []
	for c_id in clusters_dic:
		table_dic = clusters_dic[c_id]['info_table']
		tweet_texts = table_dic['text']['filtered text']
		document = ''
		for text in tweet_texts: # concatenate tweets
			document += text + ' '
		corpus.append(document)
	return corpus

def tfidf(corpus,max_keywords=20):
	# from sklearn.feature_extraction.text import TfidfVectorizer
	corpus_len = len(corpus)
	vectorizer = TfidfVectorizer(max_df=corpus_len//2)
	X = vectorizer.fit_transform(corpus)
	keywords_dic = {}
	for idx in range(corpus_len):
		# place tf-idf values in a pandas data frame
		df = pd.DataFrame(X[idx].T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"])
		df = df.sort_values(by=["tfidf"],ascending=False).head(max_keywords)
		keywords_dic[idx] = df.reset_index().rename(columns={'index':'keyword'})
	return keywords_dic



#############################################################
## Functions for Community data (comparing clusters)
#############################################################

def community_data(G):
	# get the hashtags for each community and inter-communities
	# return them in dics of dics 
	# import ast
	tags_dic = {}
	dates_dic = {}
	url_dic = {}
	text_dic = {}
	for node1,node2,data in G.edges(data=True):
		if node1 == node2:
			print('Self edge',node1)
		n1_com = G.nodes[node1]['community']
		n2_com = G.nodes[node2]['community']
		# Convert string to list
		#x = ast.literal_eval(data['hashtags'])
		#d = ast.literal_eval(data['date'])
		#u = ast.literal_eval(data['urls'])
		#keywords = [n.strip() for n in x]
		#date_list = [n.strip() for n in d]
		#urls = [n.strip() for n in u]
		keywords = json.loads(data['hashtags'])
		date_list = json.loads(data['date'])
		urls = json.loads(data['urls'])
		texts = json.loads(data['text'])

		# fill the dics of dics
		if n1_com not in tags_dic:
			tags_dic[n1_com] = {}
			dates_dic[n1_com] = {}
			url_dic[n1_com] = {}
			text_dic[n1_com] = {}
		if n2_com not in tags_dic[n1_com]:
			tags_dic[n1_com][n2_com] = keywords
			dates_dic[n1_com][n2_com] = date_list
			url_dic[n1_com][n2_com] = urls
			text_dic[n1_com][n2_com] = texts
		else:
			tags_dic[n1_com][n2_com] += keywords 
			dates_dic[n1_com][n2_com] += date_list
			url_dic[n1_com][n2_com] += urls
			text_dic[n1_com][n2_com] += texts
	return tags_dic, dates_dic, url_dic,text_dic

def compute_meantime(date_list):
	# return mean time and standard deviation of a list of dates in days
	# import numpy as np
	d_list = [ datetime.strptime(dt,'%Y-%m-%d %H:%M:%S') for dt in date_list]
	second_list = [x.timestamp() for x in d_list]
	meand = np.mean(second_list)
	stdd = np.std(second_list)
	return datetime.fromtimestamp(meand), timedelta(seconds=stdd)

def communities_date_hashtags(dates_dic, tags_dic):
	# Create a table with time and popular hashtags for each community
	# from collections import Counter
	comm_list = []
	nb_partitions = len(tags_dic.keys())
	for key in range(nb_partitions):
		most_common = Counter(tags_dic[key][key]).most_common(5)
		meandate,stddate = compute_meantime(dates_dic[key][key])
		#print('Community',key)
		#print(most_common)
		#print('Average date: {} and std deviation: {} days'.format(meandate.date(),stddate.days))
		comm_dic = {'Community':key, 'Average date':meandate.date(), 'Deviation (days)':stddate.days}
		for htag_nb in range(5): # filling the table with the hashtags
			if htag_nb < len(most_common):
				comm_dic['hashtag'+str(htag_nb)] = most_common[htag_nb][0]
			else:
				comm_dic['hashtag'+str(htag_nb)] = ''
		comm_list.append(comm_dic)
	community_table = pd.DataFrame(comm_list)
	return community_table

### Handling urls

def get_urls(url_df):
	# Dataframe with the urls of each cluster
	urltocomm = []
	for index_c, row in url_df.iterrows():
		for url in row['urls']:
			urltocomm.append([url,index_c,1])
	url_table = pd.DataFrame(urltocomm, columns=['url','Community','Occurence'])
	url_table = url_table.groupby(['url','Community']).agg(Occurence=('Occurence',sum))
	url_table = url_table.reset_index()
	return url_table

def communities_urls(url_dic):
	# Dataframe with the urls of each cluster
	urltocomm = []
	for key in url_dic:
		for url in url_dic[key][key]:
			urltocomm.append([url,key,1])
	url_table = pd.DataFrame(urltocomm, columns=['url','Community','Occurence'])
	url_table = url_table.groupby(['url','Community']).agg(Occurence=('Occurence',sum))
	url_table = url_table.reset_index()
	return url_table

def convert_bitly_table(url_table):
	# create a dataframe of correspondances for bit.ly urls 
	import requests

	session = requests.Session()  # so connections are recycled

	for index, row in url_table.iterrows():
		url = row['url']
		if 'bit.ly' in url:
			try:
				resp = session.head(url, allow_redirects=True)
				url_table.loc[index,'url_c'] = resp.url
			except requests.exceptions.RequestException as e:  # This is the correct syntax
				print(' exception raised for url',url)
				print(e)
				url_table.loc[index,'url_c'] = url
		else:
			url_table.loc[index,'url_c'] = url
	return url_table


def drop_twitter_urls(url_table):
	# Drop the references to twitter web site
	if url_table.empty:
		return url_table
	twitterrowindices = url_table[url_table['url_c'].str.contains('twitter.com')].index
	return url_table.drop(twitterrowindices)


#############################################################
## Functions for managing twitter accounts to follow
#############################################################

class initial_accounts:
	""" Handle the initial twitter accounts (load ans save them)
	"""
	accounts_file = 'initial_accounts.txt' # Default account file
	accounts_dic = {}
	
	def __init__(self,accounts_file=None):
		if accounts_file is not None:
			self.accounts_file = accounts_file
		self.load()

	def accounts(self,label=None):
		#if not self.accounts_dic:
		#	self.load()
		if label is None:
			return self.accounts_dic
		self.check_label(label)
		return self.accounts_dic[label]

	def list(self):
		return list(self.accounts_dic.keys())

	def add(self,label,list_of_accounts):
		self.accounts_dic[label] = list_of_accounts

	def remove(self,label):
		self.check_label(label)
		del self.accounts_dic[name]

	def save(self):
		with open(self.accounts_file, 'w') as outfile:
			json.dump(self.accounts_dic, outfile)
		print('Wrote',self.accounts_file)

	def load(self):
		with open(self.accounts_file) as json_file:
			self.accounts_dic = json.load(json_file)

	def check_label(self,label):
		if label not in self.accounts_dic:
			print('ERROR. Key "{}" is not in the list of accounts.'.format(label))
			print('Possible choices are: {}'.format([key for key in self.accounts_dic.keys()]))
			raise keyError
	
####################################################################
# Utils
####################################################################

def initialize_folder(path_folder_list):
	# create or clean the path
	# import os
	folder_concat = ''
	for folder in path_folder_list[:-1]:
		folder_concat += folder + '/'
		if not os.path.isdir(folder_concat):
			os.mkdir(folder_concat)
			print('Path created:',folder_concat)
	# Special treatment for the last folder
	folder_concat += path_folder_list[-1] + '/'
	if not os.path.isdir(folder_concat):
		os.mkdir(folder_concat)
		print('Path created:',folder_concat)
	else:
		for f in os.listdir(folder_concat):
			os.remove(os.path.join(folder_concat, f))
		print('Cleaned path',folder_concat)
	return folder_concat


def convert_bitly_url(url_string):
	# return the true URL from the bit.ly one
	import requests

	session = requests.Session()  # so connections are recycled

	if 'bit.ly' in url_string:
		try:
			resp = session.head(url_string, allow_redirects=True)
			return resp.url
		except requests.exceptions.RequestException as e:  # This is the correct syntax
			print(' exception raised for url',url)
			print(e)
			return url_string
	return url_string

