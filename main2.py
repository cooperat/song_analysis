from __future__ import print_function
from pandas import read_csv
from pandas import DataFrame
import pandas
from utils.text_analysis import get_tfidf, get_lemmatized_sentence, get_count
from utils.clustering import cluster_poems_per_text
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


def preprocess_DB(songs_DB_path) :
    df = read_csv(songs_DB_path)

    #  general information on the dataset
    print(f'Data shape : {df.shape} \n'
        f'Columns : {df.columns} \n'
        f'Any missing information ? {(lambda x : "yes" if str(x)==True else "No")(df.isnull().any().any())}\n')

    #  statistics on the artists
    print('Statistics on the artists : \n', df['artist'].describe(), '\n')

    #  Delete duplicate songs and filter out artists with too few songs
    df['dup']=df['artist']+df['text'].str[:40]
    df.drop_duplicates(subset='dup', inplace = True)
    art_df = df.groupby('artist')['song'].count()
    to_exclude = list(art_df[art_df<10].index)
    df = df[~df['artist'].isin(to_exclude)]

    #  turn texts into lemmatized sentences and save into new csv
    print('\nStarting lemmatization\n')
    lemmatized_df = DataFrame(None, columns=['artist', 'song', 'lyrics'])
    for i in range(len(df['song'])):
        #print(df.iloc[i]['artist'], df.iloc[i]['song'])
        lemmatized_df = pandas.concat([lemmatized_df, DataFrame({'artist': [df.iloc[i]['artist']],
                                                'song': [df.iloc[i]['song']],
                                                'lyrics': [get_lemmatized_sentence(df.iloc[i]['text'])]})])
        if i % 100 == 0 :
            print("Chanson {}/{}".format(i,len(df['song'])))
    lemmatized_df.to_csv('data/songlemmatized.csv', index=False, encoding='utf8')


def cluster_songs(lemmatized_songs_path, cluster_list= [3, 5, 8, 10, 15, 20, 30, 50], seed=10):
    #  create clusters based on the KMeans algorithm
    lemmatized_df = read_csv(lemmatized_songs_path)
    texts = lemmatized_df['lyrics']
    texts = texts.values.tolist()

    for idx, i in enumerate(cluster_list):
        print("\nStarting clustering with : {} clusters ({}/{})".format(i, idx, len(cluster_list)))
        print("--------------------------")
        lemmatized_df['cluster_'+str(i)] = pandas.Series(cluster_poems_per_text(texts, i, seed), index=lemmatized_df.index)
        print("--------------------------")
    lemmatized_df.to_csv('data/songsclustered.csv', index=False, encoding='utf8')

if __name__=="__main__":
    #preprocess_DB('./data/songdata.csv')
    cluster_songs('./data/songlemmatized.csv')

        