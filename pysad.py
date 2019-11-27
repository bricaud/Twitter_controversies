# Module for #sad project

import pandas as pd
import json
import time

def fill_retweet_info(tweet_dic,raw_retweet):
	# handle the particular structure of a retweet to get the full text retweeted
    tweet_dic['retweeted_from'].append(raw_retweet['user']['screen_name'])
    if raw_retweet['truncated']:
        full_text = raw_retweet['extended_tweet']['full_text']
    else:
        full_text = raw_retweet['full_text']
    return tweet_dic, full_text

def get_user_tweets(tweet_handle, username,count=200):
    # Collect tweets from a username
    tweets_dic = {'user': [], 'date': [], 'text': [], 'favorite_count': [], 'retweet_count': [],
        'user_mentions': [], 'urls': [], 'hashtags': [], 'place': [], 'retweeted_from': []}

    for raw_tweet in tweet_handle.get_user_timeline(screen_name = username,  
                                           count = count, include_rts = True, tweet_mode='extended'):
        # Meta data
        tweets_dic['user'].append(raw_tweet['user']['screen_name'])
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(raw_tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y'))
        tweets_dic['date'].append(ts)
        tweets_dic['favorite_count'].append(raw_tweet['favorite_count'])
        tweets_dic['retweet_count'].append(raw_tweet['retweet_count'])    
        tweets_dic['user_mentions'].append([user['screen_name'] for user in raw_tweet['entities']['user_mentions']])
        tweets_dic['urls'].append([url['url'] for url in raw_tweet['entities']['urls']])
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
    row_list = []
    for idx,tweet in tweet_df.iterrows():
        user = tweet['user']
        mentions = tweet['user_mentions']
        hashtags = tweet['hashtags']
        tweet_date = [tweet['date']]
        for m in mentions:
            row_list.append({'user':user,'mention': m, 'weight': 1, 'hashtags': hashtags, 'date': tweet_date})
    mention_df = pd.DataFrame(row_list)
    if mention_df.empty:
        return None
    # this agg only works with pandas version >= 0.25
    mention_grouped = mention_df.groupby(['user','mention']).agg(weight=('weight',sum),
                                                                 hashtags=('hashtags', sum),
                                                                 date=('date', sum))
                                                                 #,date=('date',lambda x: mean(x)))#lambda x: list(x)))    
    mention_g_list = mention_df.groupby(['user','mention'])['hashtags'].apply(list)
    mention_grouped.reset_index(level=['user', 'mention'], inplace=True)
    return mention_grouped,mention_g_list

def collect_user_mention(username,python_tweets,data_path):
	# Return the mentions of a users from its tweets, together with the hashtags of the tweet where the mention is
    tweets_dic = get_user_tweets(python_tweets,username,count=200)
    tweet_df = pd.DataFrame(tweets_dic)
    mention_grouped,mention_g_list = get_mentions_edges(tweet_df)
    return mention_grouped, mention_g_list

def create_user_edgelist(python_tweets, data_path, username, thres=3):
    # Process the user username and its mentioned users
    # save in a file the edgelist for the user and each mentioned user

    # initial user
    print('Processing',username)
    #try:
    mention_grouped,mgl = collect_user_mention(username,python_tweets,data_path)
    #except:
    #    print('exception catched on user {} !!!!!!!!!!!!'.format(username))
    #    return
    mention_grouped.to_csv(data_path + username + '_mentions.csv')
    print('First user done')

    # Threshold for number of mentions
    print('Using threshold:',thres)

    for idx,row in mention_grouped.iterrows():
        #print('processing mention',idx)
        mention_name = row['mention']
        if row['weight'] < thres:
            continue
        try:
            mention_grouped,mgl = collect_user_mention(mention_name,python_tweets,data_path)
        except:
            print('exception catched on user {} !!!!!!!!!!!!'.format(username))
            continue
        if mention_grouped is not None:
            mentionfilename = data_path + mention_name + '_mentions' +'_t' +str(thres)+'.csv'
            print('Writing',mentionfilename)
            mention_grouped.to_csv(mentionfilename)

