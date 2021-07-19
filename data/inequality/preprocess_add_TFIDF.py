import numpy as np
import pandas as pd
import re, os, string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



def get_stopwords_list(stop_file_path):
    """load stop words """

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))


def clean_text(text):
    """Doc cleaning"""

    # Lowering text
    text = text.lower()

    # Removing punctuation
    text = "".join([c for c in text if c not in PUNCTUATION])

    # Removing whitespace and newlines
    text = re.sub('\s+', ' ', text)

    return text


def sort_coo(coo_matrix):
    """Sort a dict with highest score"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature, score
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def get_keywords(vectorizer, feature_names, doc):
    """Return top k keywords from a doc using TF-IDF method"""

    # generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform([doc])

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # extract only TOP_K_KEYWORDS
    keywords = extract_topn_from_vector(feature_names, sorted_items, TOP_K_KEYWORDS)

    return list(keywords.keys())





TOP_K_KEYWORDS = 10 # top k number of keywords to retrieve in a ranked document

stopwords = set(stopwords.words('english'))
data = pd.read_csv('HealthEquityTopicBib.tsv', sep='\t', usecols=[1,5], header=0, names=['title', 'abstract'])
# title_abstract.columns = ['title', 'abstract']
# title_abstract.to_csv('inequality.csv')
# # PaperTitle,Abstract
# # title,abstract
print('original data head')
print(data.head())
print('length: ', len(data))


# Constants
PUNCTUATION = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
# STOPWORD_PATH = "/kaggle/input/stopwords/stopwords.txt"
# PAPERS_PATH = "./inequality.csv"
#
#
#
# data = pd.read_csv(PAPERS_PATH)
# data.head()
data.dropna(subset=['abstract'], inplace=True)
print('after dropna data head')
print(data.head())
print('length: ', len(data))

' preparing data '

data['abstract'] = data['abstract'].apply(clean_text)
corpora = data['abstract'].to_list()

#load a set of stop words
# stopwords=get_stopwords_list(STOPWORD_PATH)

# Initializing TF-IDF Vectorizer with stopwords
vectorizer = TfidfVectorizer(stop_words=stopwords, smooth_idf=True, use_idf=True)

# Creating vocab with our corpora
# Exlcluding first 10 docs for testing purpose
vectorizer.fit_transform(corpora[10::])

# Storing vocab
feature_names = vectorizer.get_feature_names()

result = []
for doc in corpora:
    df = {}
    df['abstract'] = doc
    df['top_keywords'] = get_keywords(vectorizer, feature_names, doc)
    result.append(df)


final = pd.DataFrame(result).reset_index(drop=True)
data = data.reset_index(drop=True)
# final[final.index.duplicated()]
# stop
print('')
print('final head')
print(final.head())
print('final length ', len(final))
print('data length ', len(data))

final['title'] = data['title']

print(final.head())
final.to_csv('abstract_topTFIDF_title_HealthEquity.csv')

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

