import pandas as pd
#import json
from datetime import date

# Import the Twython class
from twython import Twython
import json
import pysad.collect as pc
import pysad.utils as pu

## Options
######Choose a category##############    
#category_name = 'swiss_climate_controversial'
#category_name = 'swiss_climate_regular'
category_name = 'coronavirus'
#category_name = 'french_tech_lesechos'
#category_name = 'swiss_immigration'
#category_name = 'swiss_immigration2'
#category_name = 'debat_burqa'
#####################################
# path
today = date.today()
date_string = today.strftime("%Y%m%d")
tweet_data_path_list = ['..','tweetdata', category_name, date_string]
results_data_path_list = ['..','resultsdata2', category_name, date_string]

# tweets
min_mentions = 2 # minimal number of mentions of a user to be followed
max_day_old = 3 # number max of days in the past
exploration_depth = 3 # mention of mention of mention of ... up to exploration depth
##


# Load credentials from json file
with open("twitter_credentials.json", "r") as file:
    creds = json.load(file)

# Instantiate an object
python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])


init_accounts = pu.initial_accounts()


username_list = init_accounts.accounts(category_name)

# create the path to save the experiment indexed with the date of today
print("date string =", date_string)

# Initialize folders (create or clean them if they exist)
tweet_data_path = pu.initialize_folder(tweet_data_path_list)
results_data_path = pu.initialize_folder(results_data_path_list)


total_user_list = pc.collect_tweets(username_list, tweet_data_path, python_tweets, min_mentions=min_mentions,
               max_day_old=max_day_old, exploration_depth=exploration_depth)
print('Data saved in',tweet_data_path)
print('Total number of users collected:')
print(len(total_user_list))