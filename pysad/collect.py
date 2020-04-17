
import pandas as pd


from datetime import datetime, timedelta, date
from twython import TwythonError, TwythonRateLimitError, TwythonAuthError # to check the returned API errors
#import preprocessor as tweetpre

from tqdm import tqdm



###############################################################
# Functions for extracting tweet info from the twitter API
###############################################################

def fill_retweet_info(tweet_dic,raw_retweet):
	# handle the particular structure of a retweet to get the full text retweeted
	tweet_dic['retweeted_from'] = raw_retweet['user']['screen_name']
	if raw_retweet['truncated']:
		full_text = raw_retweet['extended_tweet']['full_text']
	else:
		full_text = raw_retweet['full_text']
	return tweet_dic, full_text

def get_full_url(url_dic):
	if 'unwound' in url_dic:
		return url_dic['unwound']['url']
	return url_dic['expanded_url']

def extract_tweet_infos(raw_tweet):
	# make a dic from the json raw tweet with the needed information 

	tweet_dic = {}
	time_struct = datetime.strptime(raw_tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
	ts = time_struct.strftime('%Y-%m-%d %H:%M:%S')
	
	tweet_dic['user'] = raw_tweet['user']['screen_name']
	tweet_dic['name'] = raw_tweet['user']['name']
	tweet_dic['user_details'] = raw_tweet['user']['description']
	tweet_dic['date'] = ts
	tweet_dic['favorite_count'] = raw_tweet['favorite_count']
	tweet_dic['retweet_count'] = raw_tweet['retweet_count']  
	tweet_dic['user_mentions'] = [user['screen_name'] for user in raw_tweet['entities']['user_mentions']]
	tweet_dic['urls'] = [get_full_url(url) for url in raw_tweet['entities']['urls']]
	tweet_dic['hashtags'] = [htg['text'] for htg in raw_tweet['entities']['hashtags']]
	#if raw_tweet['entities']['hashtags']:
	#    print([htg['text'] for htg in raw_tweet['entities']['hashtags']])
	#print(raw_tweet)
	if 'place' in raw_tweet and raw_tweet['place'] != None:          
		tweet_dic['place'] = raw_tweet['place']['name']
	else:
		tweet_dic['place'] = None
	
	# Handle text and retweet data
	if raw_tweet['truncated']:
		if 'extended_tweet' not in raw_tweet:
			raise ValueError('Missing extended tweet information. Make sure you set options to get extended tweet from the API.')
		full_text = raw_tweet['extended_tweet']['full_text']
	elif 'full_text' in raw_tweet:
		full_text = raw_tweet['full_text']
	else:
		full_text = raw_tweet['text']    
	if 'retweeted_status' in raw_tweet:
		tweet_dic, full_text = fill_retweet_info(tweet_dic,raw_tweet['retweeted_status'])
	else:
		tweet_dic['retweeted_from'] = None
	tweet_dic['text'] = full_text
	return tweet_dic

def get_user_tweets(tweet_handle, username,count=100, max_day_old=None):
	# Collect tweets from a username
	tweets_dic = {'user': [], 'name': [], 'user_details':[], 'date': [], 
		'text': [], 'favorite_count': [], 'retweet_count': [],
		'user_mentions': [], 'urls': [], 'hashtags': [], 'place': [], 'retweeted_from': []}

	# Test if ok
	try:
		user_tweets = tweet_handle.get_user_timeline(screen_name = username,  
										   count = count, include_rts = True, tweet_mode='extended')
	except TwythonAuthError as e_auth:
		print('Cannot access to twitter API, authentification error. {}'.format(e_auth.error_code))
		if e_auth.error_code == 401:
			print('Unauthorized access to user {}. Skipping.'.format(username))
			return tweets_dic
		raise
	except TwythonRateLimitError as e_lim:
		print('API rate limit reached')
		print(e_lim)
		wait_time = int(e_lim.retry_after) - time.time()
		print('Retry after {} seconds.'.format(wait_time))
		print('Entring sleep mode at:',time.ctime())
		print('Waking up at:',time.ctime(e_lim.retry_after+1))
		time.sleep(wait_time + 1)
	except TwythonError as e:
		print('Twitter API returned error {} for user {}.'.format(e.error_code, username))
		return tweets_dic
	for raw_tweet in user_tweets:
		# Check if the tweet date is not too old
		time_struct = datetime.strptime(raw_tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
		if (max_day_old is not None) and (time_struct < datetime.now() - timedelta(days = max_day_old)):
			break # stop iterating on the tweet list
		# Structure the needed info in a dict
		tweet_dic = extract_tweet_infos(raw_tweet)
		tweets_dic['user'].append(tweet_dic['user'])
		tweets_dic['name'].append(tweet_dic['name'])
		tweets_dic['user_details'].append(tweet_dic['user_details'])
		tweets_dic['date'].append(tweet_dic['date'])
		tweets_dic['favorite_count'].append(tweet_dic['favorite_count'])
		tweets_dic['retweet_count'].append(tweet_dic['retweet_count'])    
		tweets_dic['user_mentions'].append(tweet_dic['user_mentions'])
		tweets_dic['urls'].append(tweet_dic['urls'])
		tweets_dic['hashtags'].append(tweet_dic['hashtags'])
		tweets_dic['place'].append(tweet_dic['place'])
		tweets_dic['retweeted_from'].append(tweet_dic['retweeted_from'])
		tweets_dic['text'].append(tweet_dic['text'])

	return tweets_dic


def get_edges(tweet_df):
	# Create the user -> mention table with their properties fom the list of tweets of a user
	
	# Some bots to be removed from the collection
	userstoremove_list = ['threader_app','threadreaderapp']

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
			if m in userstoremove_list:
				continue
			row_list.append({'user':user,'mention': m, 'weight': 1, 'hashtags': hashtags,
							'date': tweet_date, 'urls':urls, 'text':text,
							'retweet_count': retweet_count,'favorite_count': favorite_count})
	mention_df = pd.DataFrame(row_list)
	return mention_df

def get_nodes_properties(tweet_df,user):
	nb_popular_tweets = 5
	row_list = []
	# global properties
	all_hashtags = []
	for idx,tweet in tweet_df.iterrows(): 
		all_hashtags += tweet['hashtags']
	all_hashtags = [(x,all_hashtags.count(x)) for x in set(all_hashtags)]
	# Get most popular tweets of user
	tweet_df = tweet_df.sort_values(by='retweet_count',ascending=False)
	for idx,tweet in tweet_df.head(nb_popular_tweets).iterrows():
		user = tweet['user']
		name = tweet['name']
		user_details = tweet['user_details']
		hashtags = tweet['hashtags']
		tweet_date = tweet['date']
		urls = tweet['urls']
		text = tweet['text']
		retweet_count = tweet['retweet_count']
		favorite_count = tweet['favorite_count'] 
		row_list.append({'user': user, 'name': name,'user_details': user_details, 'all_hashtags': all_hashtags,
							'date': tweet_date, 'urls':urls, 'text':text, 'hashtags': hashtags,
							'retweet_count': retweet_count,'favorite_count': favorite_count})
	# If empty list of tweets
	if not row_list:
		row_list.append({'user': user, 'name': '','user_details': '', 'all_hashtags': [],
							'date': '', 'urls': [], 'text': '', 'hashtags': [],
							'retweet_count': 0,'favorite_count': 0})
	popular_tweets_df = pd.DataFrame(row_list)
	return popular_tweets_df


def collect_user_data(username,python_tweets, max_day_old):
	tweets_dic = get_user_tweets(python_tweets,username,count=200, max_day_old=max_day_old)
	#if not tweets_dic:
	#	print('User {} has an empty account.'.format(username))
	#	return pd.DataFrame(), pd.DataFrame()
	#print(tweets_dic)
	tweet_df = pd.DataFrame(tweets_dic)
	edges_df = get_edges(tweet_df)
	user_info_df = get_nodes_properties(tweet_df,username)
	return edges_df, user_info_df


def group_edges(edge_df):
	mention_grouped = edge_df.groupby(['user','mention']).agg(weight=('weight',sum))
	mention_grouped.reset_index(level=['user', 'mention'], inplace=True)
	return mention_grouped



def process_hop(python_tweets, data_path, username_list, min_mentions=3, max_day_old=None):
	""" collect the tweets and tweet info of the users in the list username_list
	"""
	#print('Collecting the tweets for the last {} days.'.format(max_day_old))
	users_dic = {'username':[], 'Nb_diff_mentions': []}
	new_users_list = []
	empty_tweets_users = []
	for user in tqdm(username_list):
		if not isinstance(user,str):
			continue
		edges_df, node_df = collect_user_data(user, python_tweets, max_day_old=max_day_old)
		# Collect mentioned users for the next hop
		# Only collect the ones mentioned more than min_mentions
		if not edges_df.empty:
			# Save to json file
			edgefilename = data_path + user + '_mentions' + '.json'
			nodefilename = data_path + user + '_userinfo' + '.json'
			edges_df.to_json(edgefilename)
			node_df.to_json(nodefilename)
			# Extract mentioned users
			edges_g = group_edges(edges_df)	
			users_connected = edges_g['mention'][edges_g['weight']>=min_mentions]
			new_users_list += users_connected.tolist()
		else: # keep track of the users with empty account
			empty_tweets_users.append(user)
	print('users with empty tweet list or no mention:',empty_tweets_users)
	return new_users_list 

#def process_hop(depth, python_tweets, data_path, username_list, min_mentions, max_day_old):
#	new_users_list = process_user_list(python_tweets, data_path, username_list, 
#													thres=min_mentions, max_day_old=max_day_old)
#	return new_users_list

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
		print('')
		print('******* Processing users at {}-hop distance *******'.format(depth))
		new_users_founds = process_hop(python_tweets, data_path, new_username_list, 
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
	
	return total_username_list
