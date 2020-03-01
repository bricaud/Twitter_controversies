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

from random import sample
import matplotlib.colors as mcolors

from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

def cluster_comparison(folder_path):
    # method used to compare the indicators of the clusters detected for a specific topic
    
    clusterList = []
    keywordList, tfIdfList = [], []
    tweetDF = pd.DataFrame(columns=['cluster_id', 'category', 'word', 'score'])
    
    for filename in os.listdir(folder_path):
        if filename.endswith('xlsx'):
            filepath = os.path.join(folder_path, filename)
            clusterDF = pd.read_excel(filepath, sheet_name=None)
            clusterDict = {'cluster_id': int(clusterDF['cluster']['cluster id'].dropna().values[0]),
                            'nb_nodes': clusterDF['indicators']['nb_nodes'].values[0],
                            'k_max': clusterDF['indicators']['k_max'].values[0],
                            'max_kcore': clusterDF['indicators']['max_kcore'].values[0],
                            'norm_kcore': clusterDF['indicators']['norm_kcore'].values[0],
                            'density': clusterDF['indicators']['density'].values[0],
                            'activity_per_edge': clusterDF['indicators']['activity_per_edge'].values[0],
                            'nb_influencers': clusterDF['indicators']['nb_influencers'].values[0],
                            'nb_enthousiasts': clusterDF['indicators']['nb anthousiasts'].values[0]}
            clusterValues = [clusterDict['nb_nodes'], clusterDict['k_max'], clusterDict['max_kcore'],
                              clusterDict['norm_kcore'], clusterDict['density'], clusterDict['activity_per_edge'],
                              clusterDict['nb_influencers'], clusterDict['nb_enthousiasts']]
            clusterList.append(clusterValues)
            
            # Process the hashtags and keyword
            keywordList = clusterDF['cluster']['keyword'].dropna().values
            tfIdfList = clusterDF['cluster']['tfidf'].dropna().values
            idList = np.ones(len(keywordList), dtype = int) * int(clusterDF['cluster']['cluster id'].dropna().values[0])
            keywordDF = pd.DataFrame({'cluster_id': idList,
                                   'category': ['keyword'] * len(keywordList),
                                   'word': keywordList,
                                   'score': tfIdfList})
            tweetDF = tweetDF.append(keywordDF, ignore_index=True)
            
            if 'hashtag' in clusterDF['cluster'].columns:
                hashtagList = clusterDF['cluster']['hashtag'].dropna().values
                countList = clusterDF['cluster']['count'].dropna().values
                idList = np.ones(len(hashtagList), dtype = int) * int(clusterDF['cluster']['cluster id'].dropna().values[0])
                hashtagDF = pd.DataFrame({'cluster_id': idList,
                                         'category': ['hashtag'] * len(hashtagList),
                                         'word': hashtagList,
                                         'score': countList})

                tweetDF = tweetDF.append(hashtagDF, ignore_index= True)
            
    nodesList, kmaxList, maxkcoreList, normkCoreList, densityList, activityList, inflList, enthList = zip(*clusterList)
    
    '''
    # Random colors selection
    baseColors = [value for key, value in dict(mcolors.TABLEAU_COLORS).items()]
    baseColors.extend([value for key, value in dict(mcolors.BASE_COLORS).items() if key is not 'w'])
    barColor = sample(baseColors, len(nodesList))
    
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    
    ax[0, 0].bar(range(len(nodesList)), nodesList, color=barColor)
    ax[0, 0].title.set_text('Number of Nodes')
    ax[0, 1].bar(range(len(kmaxList)), kmaxList, color=barColor)
    ax[0, 1].title.set_text('k_max')
    ax[0, 2].bar(range(len(maxkcoreList)), maxkcoreList, color=barColor)
    ax[0, 2].title.set_text('max k_core')
    ax[0, 3].bar(range(len(normkCoreList)), normkCoreList, color=barColor)
    ax[0, 3].title.set_text('norm k_core')
    ax[1, 0].bar(range(len(densityList)), densityList, color=barColor)
    ax[1, 0].title.set_text('Density')
    ax[1, 1].bar(range(len(activityList)), activityList, color=barColor)
    ax[1, 1].title.set_text('Activity per Edge')
    ax[1, 2].bar(range(len(inflList)), inflList, color=barColor)
    ax[1, 2].title.set_text('Number of Influencers')
    ax[1, 3].bar(range(len(enthList)), enthList, color=barColor)
    ax[1, 3].title.set_text('Number of Enthusiasts')
    '''
    
    ## INTERACTIVE GRAPH: Created by using Plotly
    
    clusterLabel = [str(i) for i in range(0, len(nodesList))]
    
    figNodes = go.Figure()
    figNodes.add_trace(go.Scatter(
        x=nodesList,
        y=clusterLabel,
        marker=dict(color="crimson", size=12),
        mode="markers",
        name="Nodes",
    ))
    
    figNodes.add_trace(go.Scatter(
        x=inflList,
        y=clusterLabel,
        marker=dict(color="sandybrown", size=12),
        mode="markers",
        name="Influencers",
    ))
    
    figNodes.add_trace(go.Scatter(
        x=enthList,
        y=clusterLabel,
        marker=dict(color="midnightblue", size=12),
        mode="markers",
        name="Enthusiasts",
    ))
    
    figNodes.update_layout(title="Cluster Statistics",
                  xaxis_title="# Users",
                  yaxis_title="Cluster")
    
    figNodes.show()
    
    figK = go.Figure()
    figK.add_trace(go.Scatter(
        x=kmaxList,
        y=clusterLabel,
        marker=dict(color="crimson", size=12),
        mode="markers",
        name="k_max",
    ))
    
    figK.add_trace(go.Scatter(
        x=maxkcoreList,
        y=clusterLabel,
        marker=dict(color="sandybrown", size=12),
        mode="markers",
        name="max_kcore",
    ))
    
    figK.add_trace(go.Scatter(
        x=normkCoreList,
        y=clusterLabel,
        marker=dict(color="midnightblue", size=12),
        mode="markers",
        name="norm_kcore",
    ))
    
    figK.update_layout(title="Cluster Statistics",
                  xaxis_title="Value",
                  yaxis_title="Cluster")
    
    figK.show()
    
    # Sunburst graph
    
    figSun = px.sunburst(tweetDF, path=['cluster_id', 'category', 'word'], 
                                  values='score',
                                  maxdepth=2)
    figSun.show()
    
    

