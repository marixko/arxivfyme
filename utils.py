import pandas as pd
import numpy as np
import streamlit as st
from nltk import word_tokenize 
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
from sklearn.metrics import pairwise_distances
import dill as pickle 

nltk.download("stopwords")
stemmer = PorterStemmer()
vectorizer = TfidfVectorizer()
stpwrds = set(stopwords.words("english"))
additional_stopwords = set(('ie', 'eg', 'cf', 'etc', 'et', 'al'))
stpwrds.update(additional_stopwords)


def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]

def remove_latex(s):
    regex = r"(\$+)(?:(?!\1)[\s\S])*\1"
    subst = ""
    result = re.sub(regex, subst, s, 0, re.MULTILINE)
    return result

def remove_punctuation(s):
    s = re.sub(r'\d+', '', s) # remove numbers
    s = "".join([char.lower() for char in s if char not in string.punctuation]) # remove punctuations and convert characters to lower case
    s = re.sub('\s+', ' ', s).strip() # substitute multiple whitespace with single whitespace
    return s

def remove_linebreaks(s):
    return s.replace("\n", " ")

def tokenize(s):
    return word_tokenize(s, language="english")

def remove_stopwords(s):
    return [w for w in s if not w in stpwrds]

def stem(s):
    return " ".join([stemmer.stem(w.lower()) for w in s])

def vectorize(s):
    return vectorizer.fit_transform(s)

def lemmatizer(s):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    s = [lemmatizer.lemmatize(w.lower()) for w in s]
    return s

def clean(s):
    s = remove_latex(s)
    s = remove_punctuation(s)
    s = remove_linebreaks(s)
    s = tokenize(s)
    s = remove_stopwords(s)
    # if lemma == True and stem==True:
    #     stem = False
    # if lemma:
    #     s = lemmatizer(s)
    # if stem:
    s = stem(s)
    return s

def show_wordcloud(data, maxwords):
    cloud = WordCloud(
        background_color='white',
        max_words=maxwords,
        max_font_size=30,
        scale=3,
        random_state=1)
   
    output=cloud.generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(output)
    plt.show()
    return fig


def plot_tsne():
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect=1)
    ax.plot(X_tsne[mask_astro][:, 0], X_tsne[mask_astro][:, 1], ".", alpha=0.5, c="C0", label="Astro")
    ax.plot(X_tsne[mask_bio][:, 0], X_tsne[mask_bio][:, 1], ".", alpha=0.5, c="C1", label="Bio")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    fig.tight_layout()


def get_paper_information(paper_id:str) -> dict or str:
    url = f'https://arxiv.org/abs/{paper_id}'
    
    try:
        req = requests.get(url)
        req.raise_for_status()
    except requests.exceptions.HTTPError as err:
        return str(err)

    soup = BeautifulSoup(req.text, 'html.parser')
    content = soup.find('div', {'id':'abs'})
    
    data = {}

    data['title'] = content.find('h1', {'class': 'title mathjax'})
    data['authors'] = content.find('div', {'class':'authors'})
    data['abstract'] = content.find('blockquote', {'class', 'abstract mathjax'})

    # cleaning html
    for key, tag in data.items():
        tag.span.decompose()
        data[key] = tag.text.strip()

    data['subject'] = soup.find('div', {'class':'browse'}).find('div', {'class':'current'}).text.strip()

    return data

def give_recomm(data, vectorizer, df, n=5):
    with open('X.pickle', 'rb') as f:
        X = pickle.load(f)

    new_input = clean(data)
    new_input = vectorizer.transform([data])
    # features = vectorizer.get_feature_names()

    ndb_dist_i = pairwise_distances(X, new_input)[:, 0]
    # sort_ind_i = ndb_dist_i.argsort()
    newdf = df.copy(deep=True)
    newdf.insert(1, "dist", ndb_dist_i)
    newdf.sort_values("dist", ascending=True, inplace=True)
    # st.write(sort_ind_i)
    newdf = newdf.iloc[1:,]
    st.write(newdf["title"].head(n))

    return