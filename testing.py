import numpy as np
import pandas as pd


t = np.load('x_test.npy')
print('t')
print(t)
y_val=np.load('data/inequality/y_val.npy', allow_pickle=False, fix_imports=True,encoding='latin1')
print('y')
print(y_val)



# health = pd.read_csv('./data/inequality/HealthEquityTopicBib.tsv', sep='\t', usecols=['PaperTitle', 'Abstract'])
# print(len(health))
# social = pd.read_csv('./data/inequality/SocialInequalityTopicBib.tsv', sep='\t', usecols=['PaperTitle', 'Abstract'])
# print(len(social))
# Trophhi = pd.read_csv('./data/inequality/SocialInequalityTopicBib.tsv', sep='\t', usecols=['PaperTitle', 'Abstract'])
# print(len(Trophhi))
#
# testing
#
# inequality = pd.concat([health, social, Trophhi])
#
# inequality.to_csv('./data/inequality/inequality.csv')
#
# print(len(inequality))

# a = np.load('./data/wiki_tfidf/y_tr.npy')
#
# print(type(a))
# b = np.shape(a)
# print(b)