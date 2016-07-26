# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:08:44 2016

@author: rghiglia
"""

# Packages
import sys
import time
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sb


# Custom packages
sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND\Toolbox')
from rg_toolbox_data import df_summarize

# Generic parameters
fgsz = (5, 3)
sav = 1

# Graphic parameters
sb.set() # resets parameters
sb.set(style="whitegrid")



# ----------------------------------------------------------------------------
# 0) ACQUIRING DATA
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------------
# To load data see yelp_start_20160621_bkp

# ----------------------------------------------------------------------------
# Extract Data
# ----------------------------------------------------------------------------
# See Yelp.ptx for data structure

dnm = r'C:\Users\rghiglia\Documents\ML_ND\Yelp'

# Assign data to dataframes
# It will create 'dfs', a list of data frames containing 5 datasets
# call them with dfs['business']

import glob
fnm = glob.glob(dnm + '\\' + '*.csv')
nm_df = [f.replace(dnm,'').replace('\\yelp_academic_dataset_','').replace('.csv','') for f in fnm]
dfs = {nm: None for nm in nm_df}
for nm, f in zip(nm_df, fnm):
    t0 = time.time()
    print "Processing file '{0}' ...".format(nm)
    dfs[nm] = pd.read_csv(f, dtype=object)
    print("\t\t\t\t %1.2f s" % (time.time() - t0))
nR = {nm: 0 for nm in nm_df}    # of rows
nC = {nm: 0 for nm in nm_df}    # of columns



# ----------------------------------------------------------------------------
# Looking at reviews
# ----------------------------------------------------------------------------

df_rev = dfs['review']
df_rev.info()
df_rev['votes.funny'].unique()
df_smry = df_summarize(df_rev)
print df_smry # takes a bit, but nice info!
# I guess that is how other customers view the reviews

# There could be a segmentation by country too here
# Although how would you recognize the country of the reviewer?
# Maybe by the location of the businesses that s/he reviews?

# I guess votes.funny, votes.cool, votes.useful are characteristics of the reviewer

# One obvious question could be which reviewer is more helpful?
# That should be answered already by votes.useful

# Can you extract what people find most "descriptive"? i.e. is there a
# commonality among the reviews that people find most useful?

# You could see if there is clustering in features (unsupervised) and/or
# Feature importance segmented by # of votes.useful

# Important functions in Bag Of Words representation

#In order to address this, scikit-learn provides utilities for the most common ways to extract numerical features from text content, namely:
#.) tokenizing strings and giving an integer id for each possible token, for instance by using white-spaces and punctuation as token separators.
#.) counting the occurrences of tokens in each document.
#.) normalizing and weighting with diminishing importance tokens that occur in the majority of samples / documents.

#We call vectorization the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the Bag of Words or “Bag of n-grams” representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.


# Start with pre-processing texts

from bs4 import BeautifulSoup

ex1_raw = df_rev['text'][0]
ex1_bs4 = BeautifulSoup(ex1_raw)

print ex1_raw
print ex1_bs4.get_text()
# Ok, cleaned up of HTML stuff, although it loooked like it was clean to begin with


# Punctuation and numbers
#letters_only = re.sub('[^a-zA-Z\d]', ' ', ex1_bs4.get_text()) # if you want to keep numbers
letters_only = re.sub('[^a-zA-Z\d]', ' ', ex1_bs4.get_text())


## Alternative
#import string
#from string import maketrans
#no_punctuation = ex1_bs4.get_text()
#no_punctuation.translate(maketrans('', string.punctuation))
## Error: maketrans arguments must have same length
# Better use re

print letters_only
lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words
print words

# Alternative to splitting
import nltk
from collections import Counter
tokens = nltk.word_tokenize(lower_case)
count = Counter(tokens)

# Stop words (words with little particular meaning)
from nltk.corpus import stopwords
print stopwords.words('english')
filtered = [w for w in tokens if not w in stopwords.words('english')]
count = Counter(filtered)
print count.most_common(100)


# Stemming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

stemmed = stem_tokens(filtered, stemmer)
count = Counter(stemmed)
print count.most_common(100)

# Great

'''
# Inverse frequency (if needed)
import nltk
import string
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

path = '/opt/datacourse/data/parts'
token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        shakes = open(file_path, 'r')
        text = shakes.read()
        lowers = text.lower()
        no_punctuation = lowers.translate(None, string.punctuation)
        token_dict[file] = no_punctuation
        
#this can take some time
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())
'''



