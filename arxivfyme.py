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
# st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("The [arXiv](https://arxiv.org/) is one of the best open-science platform today. It collects and serves about 15 000 new papers per day across all STEM fields. It contains about 2 million scientific publications today. Knowing and reading relevant literature is critical to any scientist's research. However, with the current enormous rate of publications, it is challenging for any scientists to keep up, find what is relevant closer to their interests. It results in sereval very inefficient aspects in the day-to-day work of scientists. The worst possible is when one finds a decade-old paper solving a problem after solving it themselves. There is an opportunity to help our community. Scientific papers contain domain specific words and language that are hard to search using general engines (e.g. Google, Bing, etc.). Domain specific engines exists (e.g. ADS) but their recommendations suffers from using author and citation networks. However, this procedure often leads to a biased view of the research on a given topic, commonly limited to recent work or close network of colleagues. Our goal is to provide a recommendation tool that helps preserve fairness and could help identify more representative research around a problem.")

stemmer = PorterStemmer()
vectorizer = TfidfVectorizer()
stpwrds = set(stopwords.words("english"))
additional_stopwords = set(('ie', 'eg', 'cf', 'etc', 'et', 'al'))
stpwrds.update(additional_stopwords)

# # read with pandas
# df_pandas = pd.read_json('arxivData.json')

# # convert string to python object
# for key in ["author", "link", "tag"]:
#     df_pandas[key] = df_pandas[key].agg(eval, axis=0)

# df_pandas.head()

# tokens = df_pandas["summary"].agg(clean)
# df_pandas["tokens"] = tokens
# df_pandas['tokens_str'] = df_pandas['tokens'].apply(lambda x: ','.join(map(str, x)))
# text = " ".join(summ for summ in df_pandas.tokens_str.astype(str))

df = pd.read_json('astro_ph_2022.json')
tokens = df["abstract"].agg(clean)
df["tokens"] = tokens
df['tokens_str'] = df['tokens'].apply(lambda x: ','.join(map(str, x)))
text = " ".join(summ for summ in df.tokens_str.astype(str))

fig = show_wordcloud(text, st.slider('max_words', 5, 500, 200, step = 10))
st.pyplot(fig)

