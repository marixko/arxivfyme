from re import I
import pandas as pd
import streamlit as st
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import streamlit as st
from utils import cleanv2, show_wordcloud, get_paper_information, give_recomm  
from PIL import Image
import dill as pickle

nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

st.title('arXivfy me')
# st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("The [arXiv](https://arxiv.org/) is one of the best open-science platform today. It collects and serves about 15 000 new papers per day across all STEM fields. It contains about 2 million scientific publications today. Knowing and reading relevant literature is critical to any scientist's research. However, with the current enormous rate of publications, it is challenging for any scientists to keep up, find what is relevant closer to their interests. It results in sereval very inefficient aspects in the day-to-day work of scientists. The worst possible is when one finds a decade-old paper solving a problem after solving it themselves. There is an opportunity to help our community. Scientific papers contain domain specific words and language that are hard to search using general engines (e.g. Google, Bing, etc.). Domain specific engines exists (e.g. ADS) but their recommendations suffers from using author and citation networks. However, this procedure often leads to a biased view of the research on a given topic, commonly limited to recent work or close network of colleagues. Our goal is to provide a recommendation tool that helps preserve fairness and could help identify more representative research around a problem.")
image = Image.open('logo.png')
st.image(image)

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

# tokens = df_pandas["summary"].agg(clean,lemma=True, stem=False)
# df_pandas["tokens"] = tokens
# df_pandas['tokens_str'] = df_pandas['tokens'].apply(lambda x: ','.join(map(str, x)))
# text = " ".join(summ for summ in df_pandas.tokens_str.astype(str))

# df = pd.read_json('astro_ph_2022.json')



df_astro = pd.read_json("astro_ph_2022.json", dtype={'id': 'string'}) #[:N_max]
df_bio = pd.read_json("q_bio_2022.json", dtype={'id': 'string'})
df = pd.concat([df_astro, df_bio])
df.reset_index(inplace=True)


# X = vectorizer.fit_transform(tokens)
# features = vectorizer.get_feature_names()

# vectorizer = pickle.load(open("vectorizer.pickle", "wb"))

with open('vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)


st.header("How does it work?")
st.markdown("Let's use https://arxiv.org/abs/2207.00322 as an example. The ID for this paper is 2207.00322. Let's check the 5 most recommended articles based on our entry.")
data = get_paper_information("2207.00322")
give_recomm(data["abstract"], vectorizer, df, 5)


st.header("See what arXivfyMe recommends you today!")

st.markdown("Based on any article, this app will check what are the most recommended articles for you to check out.")
id = st.text_input("Write down an arXiv's ID (e.g. it can be one of your published articles or one that you really like):")
n = st.sidebar.slider("Number of recommendations", 0,50,5)
data = get_paper_information(id)
output = give_recomm(data["abstract"], vectorizer,df, n )


st.header("Wordcloud")
st.write("Check the wordcloud of your recommendations")
tokens = output["abstract"].agg(cleanv2)
output["tokens"] = tokens
output['tokens_str'] = output['tokens'].apply(lambda x: ','.join(map(str, x)))
text = " ".join(summ for summ in output.tokens_str.astype(str))

fig = show_wordcloud(text, st.slider('max_words', 10, 200, 50, step = 10))
st.pyplot(fig)