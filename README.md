# Tools for collecting tweets and analyzing them

The notebook `tweeto_graph_v1.ipynb` is the main file. It collects and processes the tweets.

The module `pysad.py` contains all the functions needed for the analysis.

Don't forget to add your credential in the `json` file. an empty model is given with `twitter_credentials_empty.json`. You have to apply for a [developer account](https://developer.twitter.com/en/apply-for-access).

The file `initial_accounts.txt` contains the list of initial users from where the exploration starts, for different topics.

The notebook `add_initial_accounts.ipynb` give example on how to add new list of users to explore for the `initial_accounts.txt`.

Python scripts can automatically collect tweets `collect_tweets.py`, and process them `process_tweets.py`.

## TODO

[x] Reorganize the module `pysad` in submodules

[ ] Find better indicators of the cluster struture

[ ] Write a notebook that compares the indicators of the different clusters, displaying hashtags and keywords as well.

[ ] Find a better way to visualize the data (better than Excel files). An interactive graph?

