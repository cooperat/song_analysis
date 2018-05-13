from __future__ import print_function
from pandas import read_csv
from pandas import DataFrame
import pandas
from utils.text_analysis import get_tfidf, get_lemmatized_sentence, get_count
from utils.clustering import cluster_poems_per_text
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

df = read_csv('./data/songdata.csv')

#  general information on the dataset
print(f'Data shape : {df.shape} \n'
      f'Columns : {df.columns} \n'
      f'Any missing information ? {(lambda x : "yes" if str(x)==True else "No")(df.isnull().any().any())}\n')

#  statistics on the artists
print('Statistics on the artists : \n', df['artist'].describe(), '\n')

''''#  uncomment to turn texts into lemmatized sentences and save into new csv
lemmatized_df = DataFrame(None, columns=['artist', 'song', 'lyrics'])
for i in range(len(df['song'])):
    print(df.iloc[i]['artist'], df.iloc[i]['song'])
    lemmatized_df = pandas.concat([lemmatized_df, DataFrame({'artist': [df.iloc[i]['artist']],
                                            'song': [df.iloc[i]['song']],
                                            'lyrics': [get_lemmatized_sentence(df.iloc[i]['text'])]})])


lemmatized_df.to_csv('songlemmatized.csv', index=False, encoding='utf8')


#  lemmatize text and create clusters based on the KMeans algorithm
lemmatized_df = read_csv('./data/songlemmatized.csv')
texts = lemmatized_df['lyrics']
texts = texts.values.tolist()

for i in [3, 5, 8, 10, 15, 20, 30, 50]:
    lemmatized_df['cluster_'+str(i)] = pandas.Series(cluster_poems_per_text(texts, i), index=lemmatized_df.index)
lemmatized_df.to_csv('songsclustered.csv', index=False, encoding='utf8')
'''
lemmatized_df = read_csv('./data/songsclustered.csv')

#  get statistics on the clusters
for i in [3, 5, 8, 10, 15, 20, 30, 50]:
    grouped = lemmatized_df.groupby(['cluster_'+str(i)])
    clusters_distrib = grouped.count()
    clusters_distrib['song'].plot.bar()
    plt.rcParams["figure.figsize"] = (50, 10)
    plt.show()

    #  for each cluster, display the distribution of songs
    for j in range(i):
        print('\n', 'Cluster number :', j)
        cluster = grouped.get_group(j)
        print(cluster['artist'].describe())


'''
#  Predictions on the songs : predict to whcih cluster the song belongs
lemmatized_df = read_csv('./data/songsclustered.csv')
texts = lemmatized_df['lyrics']
texts = texts.values.tolist()
tfidf = get_tfidf(texts)
ids_poems = lemmatized_df['cluster_15']
X_train, X_test, y_train, y_test = train_test_split(tfidf, ids_poems, shuffle=True, test_size=0.2)

clf_dict = {
    'random_forest': RandomForestClassifier(n_estimators=15, max_depth=None, min_samples_split=2, random_state=0),
}

predictions = []

for clf_name, clf in clf_dict.items():
    clf.fit(X_train, y_train)
    preds = pandas.Series(clf.predict(X_test))
    results = pandas.DataFrame({'preds': preds,
                                'real': y_test})
    results.to_csv('randomforest.csv', index=False)
    print(clf_name, clf.score(X_test, y_test))'''