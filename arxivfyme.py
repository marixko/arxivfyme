import pandas as pd
import streamlit as st
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import streamlit as st
from utils import clean, show_wordcloud

nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

st.title('arXivfy me')
st.set_option('deprecation.showPyplotGlobalUse', False)

stemmer = PorterStemmer()
vectorizer = TfidfVectorizer()
stpwrds = set(stopwords.words("english"))

# read with pandas
df_pandas = pd.read_json('arxivData.json')

# convert string to python object
for key in ["author", "link", "tag"]:
    df_pandas[key] = df_pandas[key].agg(eval, axis=0)

df_pandas.head()

tokens = df_pandas["summary"].agg(clean)
df_pandas["tokens"] = tokens
df_pandas['tokens_str'] = df_pandas['tokens'].apply(lambda x: ','.join(map(str, x)))
text = " ".join(summ for summ in df_pandas.tokens_str.astype(str))

fig = show_wordcloud(text, st.slider('max_words', 5, 500, 200, step = 10))
st.pyplot(fig)