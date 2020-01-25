# Module for #sad project

import pandas as pd
import json
from datetime import datetime, timedelta, date
from twython import TwythonError
import ast
from collections import Counter
import numpy as np
import networkx as nx
import community

from tqdm import tqdm

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

def get_mentions_edges(tweet_df):
	# return the mentions summed over all the tweets in tweet_df
	# it gives a table with user,mention, number of mention
	# return the hashtags of the mentions in a separate list
	
	#mention_df = pd.DataFrame(columns=['user','mention','weight'])
	usertoremove_list = ['threader_app','threadreaderapp']
	row_list = []
	for idx,tweet in tweet_df.iterrows():
		user = tweet['user']
		mentions = tweet['user_mentions']
		hashtags = tweet['hashtags']
		tweet_date = [tweet['date']]
		urls = tweet['urls']
		text = tweet['text'] 
		for m in mentions:
			if m == user: # skip self-mentions
				continue
			if m in usertoremove_list:
				continue
			row_list.append({'user':user,'mention': m, 'weight': 1, 'hashtags': hashtags,
			 'date': tweet_date, 'urls':urls, 'text':[text]})
	mention_df = pd.DataFrame(row_list)
	if mention_df.empty:
		return pd.DataFrame(),pd.DataFrame()
	# this agg only works with pandas version >= 0.25
	mention_grouped = mention_df.groupby(['user','mention']).agg(weight=('weight',sum),
																 hashtags=('hashtags', sum),
																 date=('date', sum),
																 urls=('urls', sum),
																 text=('text', sum))
																 #,date=('date',lambda x: mean(x)))#lambda x: list(x)))    
	# TODO Check if mention_g_list is necessary
	mention_g_list = mention_df.groupby(['user','mention'])['hashtags'].apply(list)
	mention_grouped.reset_index(level=['user', 'mention'], inplace=True)
	return mention_grouped,mention_g_list

def collect_user_mention(username,python_tweets,data_path, max_day_old):
	# Return the mentions of a users from its tweets, together with the hashtags of the tweet where the mention is
	#print('Collect the Tweets of the last {} days.'.format(max_day_old))
	tweets_dic = get_user_tweets(python_tweets,username,count=100, max_day_old=max_day_old)
	if not tweets_dic:
		print('User {} has an empty tweet list.'.format(username))
		return pd.DataFrame(),pd.DataFrame()
	#print(tweets_dic)
	tweet_df = pd.DataFrame(tweets_dic)
	mention_grouped,mention_g_list = get_mentions_edges(tweet_df)
	return mention_grouped, mention_g_list

# def create_user_edgelist(python_tweets, data_path, username, thres=3, max_day_old=None):
# 	# Process the user username and its mentioned users
# 	# save in a file the edgelist for the user and each mentioned user

# 	# initial user
# 	print('Processing',username)
# 	#try:
# 	mention_grouped,mgl = collect_user_mention(username,python_tweets,data_path, max_day_old=max_day_old)
# 	#except:
# 	#    print('exception catched on user {} !!!!!!!!!!!!'.format(username))
# 	#    return
# 	if mention_grouped.empty:
# 		print('Empty tweet list. Processing stopped for user ',username)
# 		return 0,0
# 	mention_grouped.to_csv(data_path + username + '_mentions.csv')
# 	nb_mentions = len(mention_grouped)
# 	print('First user done. Nb different mentions:',nb_mentions)

# 	# Threshold for number of mentions
# 	print('Using threshold:',thres)

# 	mentions_of_mentions = 0
# 	for idx,row in mention_grouped.iterrows():
# 		#print('processing mention',idx)
# 		mention_name = row['mention']
# 		if row['weight'] < thres:
# 			continue
# 		try:
# 			mention_grouped,mgl = collect_user_mention(mention_name,python_tweets,data_path,max_day_old)
# 		except:
# 			print('exception catched on user {} !!!!!!!!!!!!'.format(username))
# 			continue
# 		if mention_grouped.empty:
# 			print('Empty tweet list. Processing stopped for user ',username)
# 			continue
# 		else:
# 			mentionfilename = data_path + mention_name + '_mentions' +'_t' +str(thres)+'.csv'
# 			print('Writing {} tweets in {}.'.format(len(mention_grouped),mentionfilename))
# 			mention_grouped.to_csv(mentionfilename)
# 			mentions_of_mentions += len(mention_grouped)
# 	return nb_mentions,mentions_of_mentions

# def remove_particular_users(users_df):
# 	userlist = ['threader_app','threadreaderapp']
# 	user_present = [user for user in userlist if user in users_df['mention']]
# 	if len(user_present)>0:
# 		print('dropping',user_present)
# 	return users_df.drop(user_present)

def create_user_edgelist_new(python_tweets, data_path, username, thres, max_day_old):
	# Process the user username and its mentioned users
	# save in a file the edgelist for the user and each mentioned user

	#print('Processing',username)
	#try:
	mention_grouped,mgl = collect_user_mention(username,python_tweets,data_path, max_day_old=max_day_old)
	#except:
	#    print('exception catched on user {} !!!!!!!!!!!!'.format(username))
	#    return
	if mention_grouped.empty:
		#print('Empty tweet list. Processing stopped for user ',username)
		return mention_grouped
	#mention_grouped = remove_particular_users(mention_grouped)
	mentionfilename = data_path + username + '_mentions' +'_t' +str(thres)+'.json'
	#print('Writing {} tweets in {}.'.format(len(mention_grouped),mentionfilename))
	mention_grouped.to_json(mentionfilename)
	#nb_mentions = len(mention_grouped)
	#print('User {} done. Nb different mentions: {}'.format(username,nb_mentions))
	return mention_grouped


def process_user_list(python_tweets, data_path, username_list, thres=3, max_day_old=None):
	""" collect the tweets and tweet info of the users in the list username_list
	"""
	users_dic = {'username':[], 'Nb_diff_mentions': []}
	print('Collecting the tweets for the last {} days.'.format(max_day_old))
	new_users_list = []
	empty_tweets_users = []
	for user in tqdm(username_list):
		mentions = create_user_edgelist_new(python_tweets, data_path, user, thres=thres, max_day_old=max_day_old)
		if not mentions.empty:
			users_mentioned = mentions['mention'][mentions['weight']>thres]
			#users_mentioned = users_mentioned.unique() # not sure this is useful
			new_users_list += users_mentioned.tolist()
		else:
			empty_tweets_users.append(user)
		users_dic['username'].append(user)
		users_dic['Nb_diff_mentions'].append(len(mentions))
	print('users with empty tweet list or no mention:',empty_tweets_users)
	users_df = pd.DataFrame(users_dic)
	return new_users_list,users_df


def collect_tweets(username_list, data_path, python_tweets, min_mentions=2, max_day_old=7, exploration_depth=4):
	""" Collect the tweets of the users and their mentions
		and save them in data_path
	"""
	print('Threshold set to {} mentions.'.format(min_mentions))
	print('Collecting the tweets for the last {} days.'.format(max_day_old))
	users_dic = {'username':[], 'Nb_mentions': [], 'mentions_of_mentions': []}
	total_username_list = username_list
	for depth in range(exploration_depth):
		print('')
		print('******* Processing users at {}-hop distance *******'.format(depth))
		new_users_list,users_df = process_user_list(python_tweets, data_path, username_list, 
													thres=min_mentions, max_day_old=max_day_old)
		#New users to collect:
		username_list = list(set(new_users_list).difference(set(total_username_list))) # remove the one already collected
		total_username_list += username_list
	print('Total number of users collected:')
	print(len(total_username_list),len(set(total_username_list)))
	return total_username_list

#############################################################
# Functions for the graph of users
#############################################################


def graph_from_edgeslist(edge_df,degree_min):
	print('Creating the graph from the edge list')
	G = nx.from_pandas_edgelist(edge_df,source='user',target='mention', edge_attr=['weight','hashtags','date','urls','text'])
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
	partition = community.best_partition(G)
	nx.set_node_attributes(G,partition,name='community')
	print('Communities saved on the graph as node attributes.')
	nb_partitions = max(partition.values())+1
	print('Nb of partitions:',nb_partitions)
	# Create a dictionary of subgraphs, one per community
	community_dic = {}
	for idx in range(nb_partitions):
		subgraph = G.subgraph([key for (key,value) in partition.items() if value==idx])
		community_dic[idx] = subgraph
	return G, community_dic

#############################################################
## Functions for cluster analysis
#############################################################

def compute_cluster_indicators(subgraph):
	gc = subgraph
	gc_size = gc.number_of_nodes()
	ck = nx.algorithms.core.core_number(gc)
	max_k = max(ck.values())
	kcurve = [len([key for (key,value) in ck.items() if value==idx]) for idx in range(max_k+1)]
	max_k_core_size = kcurve[-1]
	in_diversity = gc.number_of_edges()/gc_size
	in_activity = gc.size(weight='weight')/gc_size
	info_dic = {'nb_nodes': gc_size, 'k_max':max_k, 'max_kcore':max_k_core_size,
				'norm_kcore':max_k_core_size/gc_size,'in_diversity': in_diversity,
				'in_activity': in_activity, 'activism': in_diversity*in_activity,
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
		if node1 not in user_text:
			user_text[node1] = texts
		else:
			user_text[node1] += texts
	return user_text, hashtags, date_list, urls


def communities_date_tags_as_table(dates_list, tags_list):
	# Create a table with time and popular hashtags for each community
	# from collections import Counter
	comm_list = []
	nb_hashtags = 5
	most_common = Counter(tags_list).most_common(nb_hashtags)
	meandate,stddate = compute_meantime(dates_list)
	info_dic = {'Average date':meandate.date(), 'Deviation (days)':stddate.days}
	for htag_nb in range(nb_hashtags): # filling the table with the hashtags
		if htag_nb < len(most_common):
			info_dic['hashtag'+str(htag_nb)] = most_common[htag_nb][0]
		else:
			info_dic['hashtag'+str(htag_nb)] = ''
	return info_dic

def dates_tags_table(cluster_dic):
	comm_list = []
	for c in cluster_dic:
		gc = cluster_dic[c]
		comm_dic = {'Community': c}
		user_text, hashtags, date_list, urls = cluster_textandinfo(gc)
		info_dic = communities_date_tags_as_table(date_list, hashtags)
		comm_dic = {**comm_dic, **info_dic, 'urls': urls, 'text': user_text}
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

#############################################################
## Functions for Community data
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

def convert_bitly(url_table):
	# Replace all bit.ly urls by the correct one
	import requests

	session = requests.Session()  # so connections are recycled

	for index, row in url_table.iterrows():
		url = row['url']
		if 'bit.ly' in url:
			try:
				resp = session.head(url, allow_redirects=True)
				url_table.loc[index,'url'] = resp.url
			except requests.exceptions.RequestException as e:  # This is the correct syntax
				print(' exception raised for url',url)
				print(e)
	return url_table

def drop_twitter_urls(url_table):
	# Drop the references to twitter web site
	twitterrowindices = url_table[url_table['url'].str.contains('twitter.com')].index
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
	