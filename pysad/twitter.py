
# Import the Twython class
from twython import Twython
import json
import pandas as pd

		
from datetime import datetime, timedelta, date
from twython import TwythonError, TwythonRateLimitError, TwythonAuthError # to check the returned API errors

#import random

from tqdm import tqdm



class twitter_network:

	rules = {}


	def __init__(self,credential_file):

		# Load credentials from json file
		#"twitter_credentials.json"
		with open(credential_file, "r") as file:
			creds = json.load(file)

		# Instantiate an object
		self.twitter_handle = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
		self.rules['min_mentions'] = 0
		self.rules['max_day_old'] = None
		self.rules['max_tweets_per_user'] = 200

	def get_neighbors(self,user):
		if not isinstance(user,str):
			return pd.DataFrame(),pd.DataFrame()
		tweets_dic = self.get_user_tweets(user)
		edges_df, node_df = self.edges_nodes_from_user(tweets_dic,user)
		return edges_df,node_df

	def filter_nodes(self,nodes_df):
		return nodes_df

	def filter_edges(self,edges_df):
		edges_g = self.group_edges(edges_df)	
		users_to_remove = edges_g['mention'][edges_g['weight'] < self.rules['min_mentions']]
		# Get names of indexes for which column Age has value 30
		indexNames = edges_df[ edges_df['user'].isin(users_to_remove) ].index
		# Delete these row indexes from dataFrame
		edges_df.drop(indexNames , inplace=True)
		return edges_df

	def neighbors_list(self,edges_df):
		#print(edges_df)
		#print(edges_df['mention'].unique())
		users_connected = edges_df['mention'].unique().tolist()
		return users_connected



	###############################################################
	# Functions for extracting tweet info from the twitter API
	###############################################################

	def fill_retweet_info(self,tweet_dic,raw_retweet):
		# handle the particular structure of a retweet to get the full text retweeted
		tweet_dic['retweeted_from'] = raw_retweet['user']['screen_name']
		if raw_retweet['truncated']:
			full_text = raw_retweet['extended_tweet']['full_text']
		else:
			full_text = raw_retweet['full_text']
		return tweet_dic, full_text

	def get_full_url(self,url_dic):
		if 'unwound' in url_dic:
			return url_dic['unwound']['url']
		return url_dic['expanded_url']

	def extract_tweet_infos(self,raw_tweet):
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
		tweet_dic['urls'] = [self.get_full_url(url) for url in raw_tweet['entities']['urls']]
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
			tweet_dic, full_text = self.fill_retweet_info(tweet_dic,raw_tweet['retweeted_status'])
		else:
			tweet_dic['retweeted_from'] = None
		tweet_dic['text'] = full_text
		return tweet_dic

	def get_user_tweets(self,username):
		# Collect tweets from a username
		max_day_old = self.rules['max_day_old']
		count = self.rules['max_tweets_per_user']
		tweets_dic = {'user': [], 'name': [], 'user_details':[], 'date': [], 
			'text': [], 'favorite_count': [], 'retweet_count': [],
			'user_mentions': [], 'urls': [], 'hashtags': [], 'place': [], 'retweeted_from': []}

		# Test if ok
		try:
			user_tweets = self.twitter_handle.get_user_timeline(screen_name = username,  
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
			tweet_dic = self.extract_tweet_infos(raw_tweet)
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


	def edges_nodes_from_user(self,tweets_dic,username):
		# Make an edge and node property dataframes
		tweet_df = pd.DataFrame(tweets_dic)
		edges_df = self.get_edges(tweet_df)
		user_info_df = self.get_nodes_properties(tweet_df,username)
		return edges_df, user_info_df



	def get_edges(self,tweet_df):
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

	def get_nodes_properties(self,tweet_df,user):
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



	def group_edges(self,edge_df):
		mention_grouped = edge_df.groupby(['user','mention']).agg(weight=('weight',sum))
		mention_grouped.reset_index(level=['user', 'mention'], inplace=True)
		return mention_grouped