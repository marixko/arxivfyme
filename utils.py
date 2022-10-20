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

nltk.download("stopwords")

def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]

def remove_linebreaks(s):
    return s.replace("\n", " ")

def tokenize(s):
    return word_tokenize(s, language="english")

def remove_stopwords(s):
    stpwrds = set(stopwords.words("english"))
    return [w for w in s if not w in stpwrds]

def stem(s):
    return " ".join([stemmer.stem(w.lower()) for w in s])

def vectorize(s):
    return vectorizer.fit_transform(s)

def clean(s):
    s = re.sub(r'\d+', '', s) # remove numbers
    s = "".join([char.lower() for char in s if char not in string.punctuation]) # remove punctuations and convert characters to lower case
    s = re.sub('\s+', ' ', s).strip() # substitute multiple whitespace with single whitespace
    s = remove_linebreaks(s)
    s = tokenize(s)
    s = remove_stopwords(s)
    # s = stem(s)
    return s


def show_wordcloud(data, max_words):
    cloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)
   
    output=cloud.generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(output)
    plt.show()