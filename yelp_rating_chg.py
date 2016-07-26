# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 08:23:23 2016

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
# 1) DESCRIBING AND UNDERSTANDING THE DATA
# ----------------------------------------------------------------------------

# dfs is a list of data frames containing 5 datasets

# -----------------------------------------------------------------------------
# Business
# -----------------------------------------------------------------------------
nm0 = 'business'
df = dfs[nm0].copy()    # may be inefficient but I might change the data
nR[nm0], nC[nm0] = df.shape
print "Data '{0}' has {1} rows x {2} columns".format(nm0, nR[nm0], nC[nm0])
df_bus_smry = df_summarize(df)
print df_bus_smry
# 77k entries, with 98 columns
# This is a good way to summarize data:
# Fisrt column is the column name
# Second column is how many non-null entries
# Third column is how many distinct values it has

# Notes:
# busID is unique nB IDs
# things like full_address:
# all have an entry but not nB uniques
# since they are all different businesses, either you have two businesses 
# at the same address or there might be a default value that is not a null

# There are 9710 categories

# I want to know how many entries we have for each category
# I think that's a groupby thingy
#grp = pd.groupby(df, 'categories')
#df_tmp = grp.count()
#df_tmp.info()
# Much better implementation
grp_cat = df.groupby('categories').size().sort_values(ascending=False)
print grp_cat

# I see, so the value for category is actually a multiple thing already:
df.ix[0,'categories']

# I'd like to know how many entries we have per category
grp = pd.value_counts(df['categories'])     # type(grp) # Series
df_grp = pd.DataFrame({'cat': grp.index, 'cnt': grp.values})
print df_grp

f, ax = plt.subplots(figsize=(10,4))
n = 35
x = [i-0.25 for i in range(n)]
ax.bar(x, df_grp['cnt'].head(n), width=0.5);
ax.set_title('Count per (Joint) Category')
ax.set_xticks(range(n));
ax.set_xticklabels(df_grp['cat'].iloc[:n], rotation='vertical');
ax.set_xlim((-0.25, n));
if sav: plt.savefig(dnm + '\\' + 'Business_cat.png', bbox_inches='tight')

# df_grp['cnt'].where(df_grp['cnt']==1) # not really
ix_1 = np.where(df_grp['cnt']==1) # type(ix) # tuple
ix_1bp = np.where(df_grp['cnt']==int(0.0001*nR[nm0]))
ix_1[0][0]
ix_1bp[0][0]
# Actually you want say 99% of the corpus being described
ix_99 = np.where(df_grp['cnt'].cumsum()>=0.99*nR[nm0])
ix_99[0][0]
# You have to go pretty far into the tail to get 99% coverage


# Separate out the categories
catU = []
for cats in df_grp['cat']:
    catss = [wrd.strip() for wrd in re.split(',',re.sub('[\[\]\']', '', cats))] # removes "'" "[" and "]", then splits by comma then trims each entry in the list
    catU += catss
len(catU)
#catU = list(set(catU)).sort() # ?? NoneType
catU = (list(set(catU)))
len(catU)
catU.sort()
len(catU)

# And now?
# You'd need to re-classify them if you just want restaurants
# Do you really want that?
# Or maybe you just pick restaurants

# You really need to start focusing on some specific question

# Maybe Q1) Ratings change in Restaurants

#ixRestaurants = []
#for (i, cats) in enumerate(df_grp['cat']):
#    catss = [wrd.strip() for wrd in re.split(',',re.sub('[\[\]\']', '', cats))]
#    if 'Restaurants' in catss:
#        ixRestaurants.append(i)
## Eh no Ciccio! L'indice e' nel dataframe df_grp!
# 3865 restaurants: No
# There are 3865 joint categories whose one of the category is restaurant


ixRestaurants = []
for (i, cats) in enumerate(df['categories']):
    catss = [wrd.strip() for wrd in re.split(',',re.sub('[\[\]\']', '', cats))]
    if 'Restaurants' in catss:
        ixRestaurants.append(i)
len(ixRestaurants)
# 25,071 restaurants

df_res = df.iloc[ixRestaurants]
len(df_res)

# Ok, so this could be a subset of the dataset (including training and testing)

# One question could be:
# Ratings change for restaurants
# Bias US/EU in features: are there features that they care more/less about?
# Should translate into a supervised problem where target = nationality or continent






## Initialize the matplotlib figure
#f, ax = plt.subplots(figsize=fgsz)
#sb.set_color_codes("pastel")
#sb.barplot(x="cat", y="cnt", data=df_grp, label="Total", color="b")
## This takes a LONG time

#plt.bar()
df_grp['cnt'].plot(figsize=fgsz);
plt.title('Count per Joint Category')


# If you want to start analysing the data I think you need to split those apart

# One of the question has to do with cities. Can I get that info from somewhere?

grp_city = df.groupby('city').size().sort_values(ascending=False)
# Usual crap ...
# there might be some bad character, I can't use the variable explorer
# Then: different spelling of Montreal, maybe Mathews, etc.



#Cities:
#
#    U.K.: Edinburgh
#    Germany: Karlsruhe
#    Canada: Montreal and Waterloo
#    U.S.: Pittsburgh, Charlotte, Urbana-Champaign, Phoenix, Las Vegas, Madison

#So you might want to add country as a tag

# I think I want to stick with restaurants, so you'll need to filter them out

# The first question has to do with preferences: parking, type of food, etc.
# is there a bias US vs. EU?

# Second question has to do with location. I guess you'd try to relate business
# success with location, how do you define success? stars?

# Third question about seasonal effects. This is across categories. Could be
# interesting, but is it machine learning? I guess ML might just be: find a 
# relationship ...

# Fourth question: non-intuitive categories. Complex, might need NLP

# Fifth question: NLP. Guess review rating from text. Wow!

# Sixth question: VERY VERY IMPORTANT!!!!!!!!
#Changepoints and Events: Can you detect when things change suddenly (i.e. a business coming under new management)? Can you see when a city starts going nuts over cronuts?
# Jeez, I thought the underlying assumption is of stationarity!
# Welcome to the real world!!!!!


#Social Graph Mining: Can you figure out who the trend setters are and who found the best waffle joint before waffles were cool? How much influence does my social circle have on my business choices and my ratings?
# Also very interesting and complex!!

# All-in-all very rich! I think I will stick with this data set. There is plenty of stuff to work on

# Start with a simple thing:
# Is stuff time-varying?
# Seasonalities?
# Interests and country bias?

# Let's see reviews per busID
# Start with most common busID
# Look into Review data set


# -----------------------------------------------------------------------------
# Reviews
# -----------------------------------------------------------------------------
df = dfs['review'].copy()
df.info()
df.head()
# Discard text for now
df.drop('text', axis=1, inplace=True)
df.info()
df.head()
nn = df_summarize(df)

grp = df.groupby(['business_id', 'review_id'])
grp_s = grp.size().sort_values(ascending=False)
grp_s[0:5]

df.groupby(['business_id', 'review_id']).count()

bID_grp = df[['business_id', 'review_id']].groupby(['business_id']).count()
bID_grp.sort_values('review_id', ascending=False, inplace=True)

# Let's look at the most popular business
rec = dfs['business'][dfs['business']['business_id']==bID_grp.index[0]]

# Plot the average star rating over time
i0 = 145

revs = dfs['review'][dfs['review']['business_id']==bID_grp.index[i0]]
x = revs['stars'].astype(int)
#s = pd.Series(revs['stars'].astype(int), index=revs['date']) # doesn't work ... not sure why
s = revs['stars'].astype(int)
s.index = revs['date']
s.sort_index(axis=0, ascending=True, inplace=True)

fig = plt.figure(figsize=fgsz)
s.plot()
pd.rolling_mean(s, 100).plot(color='r')



# Time-varying nature of ratings seems quite interesting!
#
## Review
#df.info()
#
#bID_grp = df['business_id'].count()
#
##pd.value_counts(d[["col_title1","col_title2"]].values.ravel()) 
#grp = pd.value_counts(df[['business_id']].values.ravel()) # .ravel() flattens the dataframe
#grp = pd.value_counts(df['business_id'])
#
## I think what you had in mind was:
#bID_grp = df.groupby(['business_id', 'review_id']) # or
#bID_grp = df.groupby(['business_id', 'user_id', 'review_id']) # possibly same reviewer made mutliple reviews to the same place; that could be an indication of change in management or other relevant change especially if dates are farther apart
#bID_grp.count()
#
#df['stars'] = df['stars'].astype(int)
#bID_grp = df.groupby(['business_id', 'user_id'])
#stat = bID_grp['stars']
#stat.agg('mean')
#df_tmp = stat.agg('count')
#type(df_tmp) # Series with a multi-index I assume
## Can you sort it?
## Sort by most popular b_id then reviewer largest number of reviews
## Possible, will look into if necessary


# In simplest form:
grp = pd.value_counts(df['business_id'])
type(grp) # Series
df_b_id = pd.DataFrame({'b_id': grp.index, 'cnt': grp.values})
df_b_id.head()
df_b_id['cnt'].plot(figsize=fgsz)


# 7/12/2016: Converting to date objects is very slow
# I will simply choose a subset and keep it in # of reviews

# Thing is you have to control for age, so maybe you want to look at reviews per week
# There might be methods for aggregating over time

#by_week = df['date'].groupby(lambda x: x.week)
## Didn't work

## Probably need to convert date to appropriate date obect
##https://docs.python.org/2/library/datetime.html#datetime.datetime.strptime
#from datetime import datetime
#date_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
#n = len(df)
### This will take an ENORMOUS amount of time!
##for (i, d) in enumerate(df['date']):
##    print "%i %i" % (i, n)
##    df['date'].iloc[i] = datetime.strptime(d, '%Y-%m-%d')
#
#
## Still incredibly slow!
#d_tmp = set(df['date'])
#nU = len(d_tmp)
#for (i, d) in enumerate(d_tmp):
#    print "%i %i" % (i, nU)
#    d_tpp = datetime.strptime(d, '%Y-%m-%d')
#    df['date'][df['date']==d] = d_tpp
#
#
## Maybe then better to store session from here
#
## Will review tomorrow
#
## Save session
#dnm = r'C:\Users\rghiglia\Documents\ML_ND\Yelp'
#fnmO = (dnm + '\\' + 'yelp_data.pkl')
#import time
#import dill                            #pip install dill --user
#start_time = time.time()
#dill.dump_session(fnmO)
#print("--- %s seconds ---" % (time.time() - start_time))
## It seems prohibitively long :(
#
#
## Load session
#dnm = r'C:\Users\rghiglia\Documents\ML_ND\Yelp'
#import time
#import dill                            #pip install dill --user
#start_time = time.time()
#dill.load_session(fnmO)
#print("--- %s seconds ---" % (time.time() - start_time))

cnt_mx = 150
df_b_id_sh = df_b_id[df_b_id.cnt>=cnt_mx]

i0 = 6
revs = dfs['review'][dfs['review']['business_id']==df_b_id_sh.ix[i0,'b_id']]
s = revs['stars'].astype(int)

ma = 50
fig = plt.figure(figsize=fgsz)
s.plot(alpha=0.4)
pd.rolling_mean(s, ma).plot(color='r')
#plt.vlines(t_x[i], 1, 5, color='k', linewidth=2)
#plt.show()


# Detecting a shift in rating
# Split sample in two, and measure conditional mean
# Slide the transition time

n = len(df_b_id_sh)
m_r = np.zeros((n, 1))      # mean review score
d_r = np.zeros((n, 1))      # difference in mean review score
t_x = np.zeros((n, 1))      # time of shift
txp = np.zeros((n, 1))      # time of shift in %
t00 = 49
ma = 50
n_mx_tmp = 200
for i in range(n_mx_tmp):
    revs = dfs['review'][dfs['review']['business_id']==df_b_id_sh.ix[i,'b_id']]
    s = revs['stars'].astype(int)
    m1, m2 = [], []
    dn = int(len(s)) / 8
    t0 = max([t00, dn])
#    print 't0 = {}'.format(t0)
    for t in range(t0,len(s)-t0):
        m1.append(s[:t].mean())
        m2.append(s[t:-1].mean())
    m_r[i] = s.mean()
    d_r[i] = np.abs(np.array(m2)-np.array(m1)).max()
    i_x0 = s.index[np.abs(np.array(m2)-np.array(m1)).argmax()]
    t_x[i] = t0 + i_x0
    txp[i] = float(i_x0) / len(s)
    print "Business ({}/{}): m = {:1.2f}, d = {:1.2f} ".format(i, n, m_r[i][0], d_r[i][0])
    
#    fig = plt.figure(figsize=fgsz)
#    s.plot(alpha=0.4)
#    pd.rolling_mean(s, 50).plot(color='r')
#    plt.vlines(t_x[i], 1, 5, color='k', linewidth=2)
#    plt.show()


fig = plt.figure(figsize=fgsz)
plt.scatter(m_r, d_r)
plt.title("Rating's Jump vs. Mean Rating")
plt.xlabel('Mean Rating')
plt.ylabel("Rating's Jump")

# Ok, you'll let it run afterwards
# What would you do next? Clustering

# TODO: Apply your clustering algorithm of choice to the reduced data 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.DataFrame(np.array(([m_r[:, 0], d_r[:, 0]])).T, columns=['Mean', 'Jump'])
data = data.iloc[:n_mx_tmp, :]

n_clusters = [2, 3]
nC = len(n_clusters)
score = np.zeros((nC, 1))
clr = ['r', 'b', 'y', 'm', 'c', 'k']

for i, n_cl in enumerate(n_clusters):
    print "Fitting with # clusters = %i" % n_cl
    clf = KMeans(init='k-means++', n_clusters=n_cl, n_jobs=1)
    clf.fit(data)

    # TODO: Predict the cluster for each data point
    preds = clf.predict(data)

    # TODO: Find the cluster centers
    centers = clf.cluster_centers_

#    # TODO: Predict the cluster for each transformed sample data point
#    sample_preds = clf.predict(pca_samples)

    fig = plt.figure(figsize=fgsz)
    for j in range(n_cl):
        ix = preds==j
        plt.scatter(data.ix[ix,0], data.ix[ix,1], color=clr[j])
    plt.show()
    
    
    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score[i] = silhouette_score(data, clf.labels_)
    print "Score (# clusters = %i) = %1.2f" % (n_cl, score[i])

# Silhouette graph?


# Adding DBSCAN analysis
from sklearn.cluster import DBSCAN
clsDB = DBSCAN(eps=0.2, min_samples=2) # manually optimized on eps
cls_fit = clsDB.fit(data)
scoreDB = silhouette_score(data, cls_fit.labels_)
print "Score (# clusters = %i) = %1.2f" % (len(set(cls_fit.labels_)), scoreDB)

# Plot
ax = plt.subplot(111, aspect='equal')
labs = cls_fit.labels_ - cls_fit.labels_.min()
labsU = set(labs)
for j in range(len(labsU)):
    ix = labs==j
    ax.scatter(data.ix[ix,0], data.ix[ix,1], color=clr[j])


# Start thinking about the next question: cultural bias
# Explain rating (=target) via features. Careful: this problem is multi-label
# Or actually region = target and rating is one of the features
# Or is it just 'regressing' region on rating and some other features? Yeah, kind of the same as ML it


# Before you move next you might want to encapsulate some of the above, at least
# in terms of getting the data and restricting to restaurants, e.g.




