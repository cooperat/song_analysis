import pandas
from pandas import DataFrame, read_csv
import json
from pandas import read_csv
from pandas import DataFrame
import pandas
from utils.text_analysis import get_tfidf, get_lemmatized_sentence, get_count
from utils.clustering import cluster_poems_per_text
import csv
import matplotlib.pyplot as plt


with open('./data/billboard_clean.json') as f:
    bill_data = json.load(f)

bill_df = DataFrame(bill_data, columns=["year", "link_year", "title", "artist"])
bill_df = bill_df.dropna()

songs_df = read_csv('./data/songsclustered.csv')

#  get name of the artists present in both files
bill_artists, all_artists = bill_df['artist'].unique(), songs_df['artist'].str.lower().unique()
match_artists = set(bill_artists).intersection(all_artists)
#
songs_df['is_hit'] = [(songs_df['artist'][i].lower() in match_artists) for i in range(len(songs_df.index))]
songs_df.to_csv('./data/songscomplete.csv', index=False, encoding='utf8')

'''
#  get name of the songs present in both files
bill_songs, all_songs = bill_df['title'].unique(), songs_df['song'].str.lower().unique()
match_songs = set(bill_songs).intersection(all_songs)


clusters = songs_df.groupby(['cluster_3', 'is_hit'])
clusters_distrib = clusters.count()
clusters_distrib['song'].plot.bar()
plt.rcParams["figure.figsize"] = (50, 10)
plt.show()

'''