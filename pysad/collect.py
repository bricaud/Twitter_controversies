
import pandas as pd


from datetime import datetime, timedelta, date
from twython import TwythonError
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
		# Check if the tweet date is not too old
		time_struct = datetime.strptime(raw_tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
		if (max_day_old is not None) and (time_struct < datetime.now() - timedelta(days = max_day_old)):
			break # stop iterating on the tweet list
		# Structure the needed info in a dict
		tweet_dic = extract_tweet_infos(raw_tweet)
		tweets_dic['user'].append(tweet_dic['user'])
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

def create_user_edgelist_new(python_tweets, data_path, username, max_day_old):
	# Process the user username and its mentioned users
	# save in a file the edgelist for the user and each mentioned user
	mention_df = collect_user_mention(username,python_tweets,data_path, max_day_old=max_day_old)
	if mention_df.empty:
		return mention_df
	#mentionfilename = data_path + username + '_mentions' +'_t' +str(thres)+'.json'
	mentionfilename = data_path + username + '_mentions' + '.json'
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
		mentions_df = create_user_edgelist_new(python_tweets, data_path, user, max_day_old=max_day_old)
		# Collect mentioned users for the next hop
		# Only collect the ones mentioned more than the threshold thres 
		if not mentions_df.empty:
			mentions = group_edges(mentions_df)	
			users_mentioned = mentions['mention'][mentions['weight']>=thres]
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
	
	return total_username_list
