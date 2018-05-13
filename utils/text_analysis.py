import treetaggerwrapper as ttw
import pandas
from nltk.corpus import stopwords as sw
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk


def lower(words):
    return words.lower()


def tokenize(words):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(words)
    return tokens


def filter_words(tokens):
    customized_stopwords = stopwords.words('english')
    filtered_words = [w for w in tokens if not w in customized_stopwords]
    return filtered_words


def remove_punkts(sentence):
    sentence = lower(sentence)
    tokens = tokenize(sentence)
    filtered_words = filter_words(tokens)
    return " ".join(filtered_words)


def get_lemmatized_sentence(sentence):
    sentence = lower(sentence)
    #without_punkts = remove_punkts(sentence)
    tokens = tokenize(sentence)
    filtered = filter_words(tokens)
    lmtzr = WordNetLemmatizer()
    stemmed = [lmtzr.lemmatize(word) for word in filtered]
    return ' '.join(stemmed)


def get_tfidf(list_of_texts):
    tfidf_transformer = TfidfVectorizer(list_of_texts, encoding='utf-8')
    tfidf_matrix = tfidf_transformer.fit_transform(list_of_texts)
    return tfidf_matrix


def get_count(list_of_texts):
    count_vec = CountVectorizer(list_of_texts, encoding='utf-8')
    return count_vec.fit_transform(list_of_texts)


if __name__=="__main__":
    phrase = "I spent my whOle life working on this project"
    print(lower(phrase))
    token = tokenize(phrase)
    print(token)
    print(filter_words(token))
    print(get_lemmatized_sentence(phrase))
    print(get_tfidf([" This can be done by:", "Assuming that you have now downloaded t", "Do not forget to instal"]))