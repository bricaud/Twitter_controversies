
import json
import os
import requests

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
# Other utils
####################################################################

def initialize_folder(path_folder_list, erase=True):
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
	elif erase==True:
		for f in os.listdir(folder_concat):
			os.remove(os.path.join(folder_concat, f))
		print('Cleaned path',folder_concat)
	return folder_concat


def convert_bitly_url(url_string):
	# return the true URL from the bit.ly one
	#import requests

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

