import numpy as np
import pandas as pd

health = pd.read_csv('./data/inequality/HealthEquityTopicBib.tsv', sep='\t', usecols=['PaperTitle', 'Abstract'])
print(len(health))
social = pd.read_csv('./data/inequality/SocialInequalityTopicBib.tsv', sep='\t', usecols=['PaperTitle', 'Abstract'])
print(len(social))
Trophhi = pd.read_csv('./data/inequality/SocialInequalityTopicBib.tsv', sep='\t', usecols=['PaperTitle', 'Abstract'])
print(len(Trophhi))

testing

inequality = pd.concat([health, social, Trophhi])

inequality.to_csv('./data/inequality/inequality.csv')

print(len(inequality))

# a = np.load('./data/wiki_tfidf/y_tr.npy')
#
# print(type(a))
# b = np.shape(a)
# print(b)