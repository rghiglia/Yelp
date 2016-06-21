# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 08:23:23 2016

@author: rghiglia
"""


import matplotlib.pyplot as plt

# Wasn't necessary. Used winzip and now files are in JSON

# Use this to unzip
#import gzip
#
## Extract files
#dnm = r'C:\Users\rghiglia\Documents\ML_ND\Yelp'
#fnz = 'sample_submission.csv.gz'
#fnzL = dnm + '\\' + fnz
#fnmL = fnzL[:-3]  # remove the '.gz' from the filename
#
## Read from .gz
#with gzip.open(fnzL, 'rb') as in_file:
#    s = in_file.read()
#
## Store uncompressed file data from 's' variable
#with open(fnmL, 'w') as f:
#    f.write(s)






# ----------------------------------------------------------------------------
# Extract data
# ----------------------------------------------------------------------------
#dnm = r'C:\Users\rghiglia\Documents\ML_ND\Yelp'
#fnm = 'yelp_academic_dataset_checkin.json'
#fnmL = dnm + '\\' + fnm

## Trials
##import pandas as pd
##df = pd.read_json(fnmL)
### Error:
###ValueError: Trailing data
##
##import sys
##sys.path.append(dnm)
##import json_to_csv_converter
### Worked now
##
### nope: json_to_csv_converter(fnmL)
##
##fnmO = 'yelp_academic_dataset_checkin.csv'
##fnmOL = dnm + '\\' + fnmO


## List files
#import glob
#fnmO = glob.glob(dnm + '\\' + '*.json')
#
#for f in fnmO:
#    print "Processing file '{}'".format(f)
#    clmns = json_to_csv_converter.get_superset_of_column_names_from_file(f)
#    read_and_write_file(f, f.replace('.json','.csv'), clmns)


#import pandas as pd
#dnm = r'C:\Users\rghiglia\Documents\ML_ND\Yelp'
#
#
#fnm = 'yelp_academic_dataset_checkin.csv'
#df_checkin = pd.read_csv(dnm + '\\' + fnm)
## Worked!
#
#fnm = 'yelp_academic_dataset_business.csv'
#df_busines = pd.read_csv(dnm + '\\' + fnm, dtype=object)
### Error:
##C:\Users\rghiglia\Anaconda2\lib\site-packages\IPython\core\interactiveshell.py:2723: DtypeWarning: Columns (1,4,7,17,29,49,60,62,79,86,94) have mixed types. Specify dtype option on import or set low_memory=False.
##  interactivity=interactivity, compiler=compiler, result=result)
## Ok, worked
#
#fnm = 'yelp_academic_dataset_review.csv'
#df_review = pd.read_csv(dnm + '\\' + fnm)
#
#fnm = 'yelp_academic_dataset_tip.csv'
#df_tip = pd.read_csv(dnm + '\\' + fnm)
#
#fnm = 'yelp_academic_dataset_user.csv'
#df_usr = pd.read_csv(dnm + '\\' + fnm)


# DO NOT ERASE THIS PART!
import pandas as pd
dnm = r'C:\Users\rghiglia\Documents\ML_ND\Yelp'

import time
start_time = time.time()

# Assign data to dataframes
import glob
fnm = glob.glob(dnm + '\\' + '*.csv')
nm_df = [f.replace(dnm,'').replace('\\yelp_academic_dataset_','').replace('.csv','') for f in fnm]
dfs = {nm: None for nm in nm_df}
for nm, f in zip(nm_df, fnm):
    print "Processing file '{0}' ...".format(nm)
    dfs[nm] = pd.read_csv(f, dtype=object)
print("--- %s seconds ---" % (time.time() - start_time))


## Try to save data to file
#
#fnmO = (dnm + '\\' + 'yelp_data.pkl')
#import time
#import dill                            #pip install dill --user
#start_time = time.time()
#dill.dump_session(fnmO)
#print("--- %s seconds ---" % (time.time() - start_time))

## Load session
#import time
#start_time = time.time()
#dnm = r'C:\Users\rghiglia\Documents\ML_ND\Yelp'
#fnmO = (dnm + '\\' + 'yelp_data.pkl')
#import dill                            #pip install dill --user
#dill.load_session(fnmO)
#print("--- %s seconds ---" % (time.time() - start_time))
## Takes almost 10 times than reading data in ...




# See Yelp.ptx for data structure


# Now study the data
# dfs is a list of data frames


# -----------------------------------------------------------------------------
# Business
# -----------------------------------------------------------------------------
df = dfs['business']
df.info()
df.shape
df.count() # already a summary of non-nulls but the code below extract more info
# 77k entries, with 98 columns

# Reorder
# Unique business_id?
# Order by non-nulls

# This is a good way to summarize data:
# Fisrt column is the column name
# Second column is how many non-null entries
# Third column is how many distinct values it has


# Use following to import
import sys
sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND\Toolbox')
from rg_toolbox_data import df_summarize
nn = df_summarize(df)

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

# I see, so the value for category is actually a multiple thing already:
df.ix[0,'categories']

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

df = dfs['review'].copy()
df.info()
df.head()
# Discard text for now
df.drop('text', axis=1, inplace=True)
df.info()
df.head()
nn = df_summarize(df)

#grp = df.groupby(['business_id', 'review_id'])
#grp_s = grp.size().sort_values(ascending=False)
#grp_s[0:5]

#!df.groupby(['business_id', 'review_id']).count()

bID_grp = df[['business_id', 'review_id']].groupby(['business_id']).count()
bID_grp.sort_values('review_id', ascending=False, inplace=True)

# Let's look at the most popular business
rec = dfs['business'][dfs['business']['business_id']==bID_grp.index[0]]

# Plot the average star rating over time
fgsz = (5, 3)
i0 = 145

revs = dfs['review'][dfs['review']['business_id']==bID_grp.index[i0]]
if 'text' in revs.columns: revs.drop('text', axis=1, inplace=True)
x = revs['stars'].astype(int)
#s = pd.Series(revs['stars'].astype(int), index=revs['date']) # doesn't work ... not sure why
s = revs['stars'].astype(int)
s.index = revs['date']
s.sort_index(axis=0, ascending=True, inplace=True)

fig = plt.figure(figsize=fgsz)
s.plot()
pd.rolling_mean(s, 100).plot(color='r')



# Time-varying nature of ratings seems quite interesting!

# Review
df.info()

bID_grp = df['business_id'].count()
bID_grp.count()

#pd.value_counts(d[["col_title1","col_title2"]].values.ravel()) 
grp = pd.value_counts(df[['business_id']].values.ravel()) # .ravel() flattens the dataframe
grp = pd.value_counts(df['business_id'])

# I think what you had in mind was:
bID_grp = df.groupby(['business_id', 'review_id']) # or
bID_grp = df.groupby(['business_id', 'user_id', 'review_id']) # possibly same reviewer made mutliple reviews to the same place; that could be an indication of change in management or other relevant change especially if dates are farther apart
bID_grp.count()

df['stars'] = df['stars'].astype(int)
bID_grp = df.groupby(['business_id', 'user_id'])
stat = bID_grp['stars']
stat.agg('mean')
df_tmp = stat.agg('count')
type(df_tmp) # Series with a multi-index I assume
# Can you sort it?
# Sort by most popular b_id then reviewer largest number of reviews
# Possible, will look into if necessary


# In simplest form:
grp = pd.value_counts(df['business_id'])
type(grp) # Series
df_b_id = pd.DataFrame({'b_id': grp.index, 'cnt': grp.values})
df_b_id.head()
df_b_id['cnt'].plot(figsize=fgsz)

# Thing is you have to control for age, so maybe you want to look at reviews per week
# There might be methods for aggregating over time

#by_week = df['date'].groupby(lambda x: x.week)
## Didn't work

# Probably need to convert date to appropriate date obect
#https://docs.python.org/2/library/datetime.html#datetime.datetime.strptime
from datetime import datetime
date_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
n = len(df)
## This will take an ENORMOUS amount of time!
#for (i, d) in enumerate(df['date']):
#    print "%i %i" % (i, n)
#    df['date'].iloc[i] = datetime.strptime(d, '%Y-%m-%d')

d_tmp = set(df['date'])
nU = len(d_tmp)
for (i, d) in enumerate(d_tmp):
    print "%i %i" % (i, nU)
    d_tpp = datetime.strptime(d, '%Y-%m-%d')
    df['date'][df['date']==d] = d_tpp


df_tmp 
