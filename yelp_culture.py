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

## Isn't this simpler? Well, it's the same thing just with more labels
#df_num['stars'].value_counts()


# Separate out the categories
catU = []
for cats in df_grp['cat']:
    catss = [wrd.strip() for wrd in re.split(',',re.sub('[\[\]\']', '', cats))] # removes "'" "[" and "]", then splits by comma then trims each entry in the list
    catU += catss
catU = (list(set(catU)))
catU.sort()

# Extract only restaurants
ixRestaurants = []
for (i, cats) in enumerate(df['categories']):
    catss = [wrd.strip() for wrd in re.split(',',re.sub('[\[\]\']', '', cats))]
    if 'Restaurants' in catss:
        ixRestaurants.append(i)
len(ixRestaurants)
# 25,071 restaurants

df_res = df.iloc[ixRestaurants].copy()
len(df_res)

# Ok, so this could be a subset of the dataset (including training and testing)


# -----------------------------------------------------------------------------
# Clean data
# -----------------------------------------------------------------------------

# Cities
df_res['city'] = [c.title() for c in df_res['city']] # capitalize city names
cityU = list(df_res['city'].unique())
cityU.sort()
# Needs clean-up, but is it necessary?
# Skipping for now

# States
stateU = list(df_res['state'].unique())
stateU.sort()

# Looking at cities and states
df_res.ix[::10,['city', 'state']]



# -----------------------------------------------------------------------------
# Augment data: need to get countries
# -----------------------------------------------------------------------------

df_res.ix[df_res['state']=='BW','city'].unique() # Baden-Wuertenberg?, GE
df_res.ix[df_res['state']=='EDH','city'].unique() # Edinburgh, UK
df_res.ix[df_res['state']=='ELN','city'].unique() # Musselburgh, Scotland, UK
df_res.ix[df_res['state']=='FIF','city'].unique() # Scotland, UK
df_res.ix[df_res['state']=='KHL','city'].unique() # Edinburgh, UK
df_res.ix[df_res['state']=='MLN','city'].unique() # UK
df_res.ix[df_res['state']=='NW','city'].unique() # GE
df_res.ix[df_res['state']=='ON','city'].unique() # Ontario, Canada
df_res.ix[df_res['state']=='QC','city'].unique() # Quebec, Canada
df_res.ix[df_res['state']=='RP','city'].unique() # GE

df_res['country'] = 'US' # careful though: you are also mapping city=None into US; you might want to be more precise; you actually have the location coordinates, I think ...
df_res.ix[df_res['state']=='BW','country'] = 'GE'
df_res.ix[df_res['state']=='EDH','country'] = 'UK'
df_res.ix[df_res['state']=='ELN','country'] = 'UK'
df_res.ix[df_res['state']=='FIF','country'] = 'UK'
df_res.ix[df_res['state']=='KHL','country'] = 'UK'
df_res.ix[df_res['state']=='MLN','country'] = 'UK'
df_res.ix[df_res['state']=='NW','country'] = 'GE'
df_res.ix[df_res['state']=='ON','country'] = 'CA'
df_res.ix[df_res['state']=='QC','country'] = 'CA'
df_res.ix[df_res['state']=='RP','country'] = 'GE'

countryU = list(df_res['country'].unique())
countryU.sort()



# -----------------------------------------------------------------------------
# Prepare the data
# -----------------------------------------------------------------------------

df_num = df_res.copy()

# Starting point:
df_num.info()

## Note:
#<class 'pandas.core.frame.DataFrame'>
#Int64Index: 25071 entries, 0 to 77438

# .) The DF has 25,071 entries and 99 columns
# .) The index is not contiguous
# .) Some columns have 0 non-null entries. Isn't the best strategy to just get rid of them right away?
# .) Strings are classified as 'object'
# .) Other stuff is also classified as object, e.g. column 'categories' is actually a list of strings

#df_num.dropna(axis=1, inplace=True) # nope: this drops anything that has at least a NaN
#df_num.info()

df_num.dropna(axis=1, how='all', inplace=True) # Ok!
df_num.info()
# Awesome

# Better to remove columns that you don't care about right away

# Get rid of other columns
for nm in ['neighborhoods', 'full_address', 'name', 'categories', 'business_id']:
    df_num.drop(nm, axis=1, inplace=True)
df_num.info()


# Jeez!! You need to convert to numerical values!!!

import sys; sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND')
from rg_toolbox_data import data_types, cat_series2num, str2timestamp

df_supp_types = data_types(df_num)  # if variable has continuous or categorical (discrete) support; I guess categorical could be ordered or not
# This method just classifies the type of data
# It does NOT make any conversion

## Note: there are stil some type='object', why? Well I was looking at df_res
#nm = 'attributes.Hair Types Specialized In.curly'
#z = df_num.columns.copy().sort_values()
#df_num[nm].head()
## Ok, all cleaned up

#cnv_str = pd.Series(index=range(len(df_num)))
df_supp_types['conv'] = 'Not converted'
dics = {}
for i, nm in enumerate(df_num.columns):
    print "Column '{}' is of '{}' support".format(nm, df_supp_types.ix[nm, 'support'])
    if not(all(df_num[nm].isnull())):
        if df_supp_types.ix[nm, 'support']=='discrete' and df_supp_types.ix[nm, 'type']=='string':
            out_tmp = cat_series2num(df_res[nm]) # convert string to numeric level
            dics[nm] = out_tmp
            col_num = out_tmp['x_num']
            col_num.name = nm + '_num'
            df_num[nm] = col_num
            df_supp_types['conv'].iloc[i] = 'Converted from string to numeric (categorical)'
        elif (df_supp_types.ix[nm, 'subtype']=='int') and (type(df_num[nm].iloc[0])==np.str):
            df_num[nm] = df[nm].astype(float)
            df_num[nm][df_num[nm].isnull()] = df_num[nm].mean()
            df_num[nm] = df_num[nm].astype(int)
            df_supp_types['conv'].iloc[i] = 'Converted to int'
        elif (df_supp_types.ix[nm, 'subtype']=='float') and (type(df_num[nm].iloc[0])==np.str):
            df_num[nm] = df[nm].astype(float)
            df_supp_types['conv'].iloc[i] = 'Converted to float'
        elif df_supp_types.ix[nm, 'type']=='time':
            df_num[nm] = pd.to_datetime(df_num[nm])
            df_supp_types['conv'].iloc[i] = 'Converted to time'

df_num = df_num[df_num.columns.sort_values()]
df_num.info()

# Handle exceptions
for nm in ['state', 'city']:
    out_tmp = cat_series2num(df_res[nm]) # convert string to numeric level
    col_num = out_tmp['x_num']
    col_num.name = nm + '_num'
    df_num[nm] = col_num
    dics[nm] = out_tmp

df_num.info()



# Now you have to deal with NaN's
# NaN's are concentrated in the hours of operation
# For now I will substitute it with average but you could probably do better by segmenting by country or city

# Average per city
# Use groupby

grp_city = df_num.groupby('city')
cnt_byCity = grp_city.count()
avg_byCity = grp_city.mean()
std_byCity = grp_city.std()
# Damn, not cities are defined in numerical valriables, and I don't recognize them ...
# Deal with NaN's BEFORE you convert to numerical or at least generate mappings
# Ok, gather info in dics['city']['x_Rdic']: from number to city

# Damn, the grouping drops the timestamps ...
# Indeed:
#TypeError: reduction operation 'mean' not allowed for this dtype
# Then I'll do it manually and pick the most common
# Or since you know it's an hour, why don;t you convert the column to just the hour in int or float?

nm = ['hours' + '.' + d + '.' + oc for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] for oc in ['open', 'close']]

for n in nm:
    df_num[n] = [d.hour for d in df_num[n]]

grp_city = df_num.groupby('city')
cnt_byCity = grp_city.count()
avg_byCity = grp_city.mean()
std_byCity = grp_city.std()
# Actually very interesting from a data analysis point of view!
# Let's look at location for example

# Almost sure there is a better way ... but need to get going
for i_c in cnt_byCity.index:
    for n in nm:
        print 'Processing city {} for attribute {}'.format(dics['city']['x_Rdic'][i_c], n)
        df_num.ix[((df_num['city']==i_c) & (df_num[n].isnull())) , n] = avg_byCity.ix[i_c, n]



df_num.info()
# Almost ... :(
len(df_num.ix[df_num['hours.Friday.close'].isnull(), 'city'].unique())
# Woof ... spread across many cities
# Try one manually
df_num.ix[df_num['city']==1, nm[0]]
# Ahh ... the city has only 1 entry?


# Check value_counts
grp = pd.value_counts(df_num['city'])
print pd.DataFrame({'cat': grp.index, 'cnt': grp.values})
len(grp[grp==1])
# Well, yeah these include those cities tha have 1 entry but value is not null ...

# Need to do a 2-tiered thing: first try by city if fail then by country ...

# START FROM HERE

grp_state = df_num.groupby('state')
cnt_byState = grp_state.count()
avg_byState = grp_state.mean()
std_byState = grp_state.std()

for i_s in cnt_byState.index:
    for n in nm:
        print 'Processing state {} for attribute {}'.format(dics['state']['x_Rdic'][i_s], n)
        df_num.ix[((df_num['state']==i_s) & (df_num[n].isnull())), n] = avg_byState.ix[i_s, n]


df_num.info()

# Damn ... I still have 3 entries left over
df_num.ix[df_num[n].isnull()][:5]


# Maybe the state is all NaN of rthose columns
grp = pd.value_counts(df['state'])     # type(grp) # Series
df_grp = pd.DataFrame({'cat': grp.index, 'cnt': grp.values})
print df_grp
# You have 9 states with only 1 entry per state


for i_s in cnt_byState.index:
    for n in nm:
        print 'Processing state {} for attribute {}'.format(dics['state']['x_Rdic'][i_s], n)
        df_num.ix[df_num[n].isnull(), n] = df_num.ix[:, n].mean()

df_num.info()

# Finally cleaned-up

# START FROM HERE

# Look at some obvious stuff

# Rating by country
grp_country = df_num.groupby('country')
cnt_byCountry = grp_country.count()
avg_byCountry = grp_country.mean()
std_byCountry = grp_country.std()
q10_byCountry = grp_country.quantile(0.1)
q90_byCountry = grp_country.quantile(0.9)
q01_byCountry = grp_country.quantile(0.01)
q99_byCountry = grp_country.quantile(0.99)

df_country = pd.DataFrame({'avg': avg_byCountry['stars'], \
    'std': std_byCountry['stars'], \
    'se': std_byCountry['stars'] / np.sqrt(cnt_byCountry['stars']), \
    'q10': q10_byCountry['stars'], 'q90': q90_byCountry['stars'], \
    'q01': q01_byCountry['stars'], 'q99': q99_byCountry['stars']})
df_country = df_country[['avg', 'std', 'se', 'q01', 'q10', 'q90', 'q99']]
df_country = df_country.rename(dics['country']['x_Rdic'])
# Awesome!
# Result #1: US lower, is it also stricter? I.e. proportionately more 1's? Or lower scores?

# You want a matrix with rows = country, columns = % within country of given score
df_num['stars'].unique()

grp_CS = df_num.groupby(['country', 'stars'])
cnt_byCS = grp_CS['state', 'type'].count()
cnt_byCS.reset_index(inplace=True)
# Ok this does it, but I am not sure it's the best way to do it
# Didn't find anything better
cnt_byCountry['state']
cnt_byCS['Total'] = cnt_byCountry['state'] # will it match indices?
# Yeah it did but need to do earlier

cnt_byCS = grp_CS['state', 'type'].count()
cnt_byCS = cnt_byCS.join(cnt_byCountry['state'], lsuffix='country') # will it match indices?
# Awesome!
df_stats = cnt_byCS['type']/cnt_byCS['state']
df_stats = df_stats.unstack().T
df_stats = df_stats.rename(columns=dics['country']['x_Rdic'])

# Plot the crashes where alcohol was involved
sb.set_color_codes('muted')
fig = plt.figure(figsize=(10,6))
for i, c in enumerate(df_stats.columns):
    fig.add_subplot(2,2,i+1)
    sb.barplot(x=df_stats.index, y=df_stats.ix[:, c])
    plt.vlines(df_country.ix[c, 'avg'], 0, 0.35)
    plt.vlines(df_country.ix[c, 'q01'], 0, 0.35, linestyle='--')
#    plt.vlines(df_country.ix[c, 'q90'], 0, 0.35, linestyle='--') # this gets completely screwed up, not sure why, despite the numerical values being correct

# Seems fair to say that it is cultural trait



# Looking for most importnat features







# Decide output variable: stars
out_varble = 'stars'
y = df_num[out_varble]
X = df_num.drop(out_varble, axis=1)
nO = len(X)
# You cannot guarantee the exact split coming out of strat sampl. let this be approx and do the count afterwards
nTst_tmp = int(nO*0.15)
#nTrn = nO - nTst


# Split data set

## NASTY: StratSampl works in iloc units!
## P-Reshuffling: the first shuffle is just to extract the test set, not for CV
#from sklearn.cross_validation import StratifiedShuffleSplit
#sss0 = StratifiedShuffleSplit(y, 1, test_size=nTst, random_state=0)
#any(np.isnan(y))
#for ix_trn, ix_tst in sss0: _ = False:
## Is ix_trn in iloc units or ix units? I think it's in iloc units!!!
#ix_trn = y.index[ix_trn]
#X_trn, y_trn = X.ix[ix_trn], y.ix[ix_trn]
#any(y_trn.isnull()) # Ok, now
#
#y.index[:10]
#ix_tmp = ix_trn.sort()
#type(ix_tmp) # NoneType ??
#ix_tmp = np.sort(ix_trn)
#type(ix_tmp) # NoneType ??
#ix_tmp[:10]
#len(ix_trn)
#len(ix_tmp)
#type(ix_tmp)
#
#X_trn, y_trn = X.ix[ix_trn], y.ix[ix_trn]
#X_tst, y_tst = X.ix[ix_tst], y.ix[ix_tst]
#
### Careful:
##any(np.isnan(y)): False
##any(y_trn.isnull()): True
### HOW THE HELL IS THIS POSSIBLE????
##There is somwthing awfully wrong here


from sklearn.cross_validation import StratifiedShuffleSplit
sss0 = StratifiedShuffleSplit(y, 1, test_size=nTst_tmp-1, random_state=0)
for ix_trn, ix_tst in sss0:
    pass
nTrn = len(ix_trn)
nTst = len(ix_tst)
if nO!=nTrn+nTst: print "Error: training and test don't sum up"

# Re-assign for correct indexing in original data frame
ix_trn = y.index[ix_trn]
ix_tst = y.index[ix_tst]
X_trn, y_trn = X.ix[ix_trn], y.ix[ix_trn]
X_tst, y_tst = X.ix[ix_tst], y.ix[ix_tst]
any(np.isnan(y))
any(y_trn.isnull())
# Ok

# Sets for CV
pctCV_tst = 0.15 # CV test
nCV_tst_tmp = int(nTrn*pctCV_tst)

nCV = 3
sss = StratifiedShuffleSplit(y_trn, nCV, test_size=nCV_tst_tmp, random_state=1)
for i, (train_index, test_index) in enumerate(sss):
    if i==0:
        nCV_trn = len(train_index)
        nCV_tst = len(test_index)
    else:
        break

if nTrn!=nCV_trn+nCV_tst: print "Error: training and test don't sum up"

# Within sss in iloc coordinates
Ix_trn = np.zeros((nCV_trn, nCV))
Ix_tst = np.zeros((nCV_tst, nCV))
for i, (train_index, test_index) in enumerate(sss):
    Ix_trn[:,i] = y_trn.index[train_index]
    Ix_tst[:,i] = y_trn.index[test_index]
    Ix_trn[:,i] = y_trn.index[train_index]
    Ix_tst[:,i] = y_trn.index[test_index]
# Still problems!
# Finally!
Ix_trn = Ix_trn.astype(int)
Ix_tst = Ix_tst.astype(int)

# Select CV train and CV test sets
i0 = 0 # e.g. select first fold
X_CV_trn, y_CV_trn = X_trn.ix[Ix_trn[:,i0]], y_trn.ix[Ix_trn[:,i0]]
X_CV_tst, y_CV_tst = X_trn.ix[Ix_tst[:,i0]], y_trn.ix[Ix_tst[:,i0]]



# Using a tree for feature importance
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4)
#clf.fit(X_CV_trn, y_CV_trn)
## Damn ... labels can only be integers 
y_CV_trn_lbl = [str(s) for s in y_CV_trn]
clf.fit(X_CV_trn, y_CV_trn_lbl)
z = pd.Series(clf.feature_importances_, X_CV_trn.columns)
zs = z.sort_values(ascending=False)
ixsDT = zs.index
# Result is pretty clear: 4 variables, one of which is longitude ...

# Next step do it by country

# START FROM HERE
Ix_CV_trn_byCountry = {0: [], 1: [], 2: [], 3: []}
for ic in X_CV_trn['country'].unique():
    Ix_CV_trn_byCountry[ic] = X_CV_trn[X_CV_trn['country']==ic].index


Zs = {0: [], 1: [], 2: [], 3: []}
for ic in X_CV_trn['country'].unique():
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    y_CV_trn_lbl = [str(s) for s in y_CV_trn.ix[Ix_CV_trn_byCountry[ic]]]
    clf.fit(X_CV_trn.ix[Ix_CV_trn_byCountry[ic]], y_CV_trn_lbl)
    z = pd.Series(clf.feature_importances_, X_CV_trn.columns)
    Zs[ic] = z.sort_values(ascending=False)

# Fascinating
nFeatImp = 7
fig = plt.figure(figsize=(10,10))
for ic in X_CV_trn['country'].unique():
    ax = fig.add_subplot(2,2,ic+1)
    sb.barplot(x=Zs[ic].index[:nFeatImp], y=Zs[ic].iloc[:nFeatImp], ax=ax)
    ax.set_ylim((0, 0.78))
    ax.set_ylim((0, 0.78))
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.set_title(dics['country']['x_Rdic'][ic], fontsize=20, weight='bold')



## This would suggest that if you use:
#review_count
#Drive-Thru
#Takes Res
#Parking street
#Monday close
#has TV
#
#AND rating
#
#You might be able to tell the country apart


# You could start with projecting onto a 2D subpace to see the 4 (new) output labels



# Decide output variable: stars
out_varbl = 'country'
y = df_num[out_varbl]
X = df_num.drop(out_varbl, axis=1)
nO = len(X)
nTst_tmp = int(nO*0.15)

from sklearn.cross_validation import StratifiedShuffleSplit
sss0 = StratifiedShuffleSplit(y, 1, test_size=nTst-1, random_state=0)
for ix_trn, ix_tst in sss0:
    pass
nTrn = len(ix_trn)
nTst = len(ix_tst)
if nO!=nTrn+nTst: print "Error: training and test don't sum up"

# Re-assign for correct indexing in original data frame
ix_trn = y.index[ix_trn]
ix_tst = y.index[ix_tst]
X_trn, y_trn = X.ix[ix_trn], y.ix[ix_trn]
X_tst, y_tst = X.ix[ix_tst], y.ix[ix_tst]
#any(np.isnan(y))
#any(y_trn.isnull())
## Ok

# Sets for CV
pctCV_tst = 0.15 # CV test
nCV_tst_tmp = int(nTrn*pctCV_tst)

nCV = 3
sss = StratifiedShuffleSplit(y_trn, nCV, test_size=nCV_tst_tmp, random_state=1)
for i, (train_index, test_index) in enumerate(sss):
    if i==0:
        nCV_trn = len(train_index)
        nCV_tst = len(test_index)
    else:
        break

if nTrn!=nCV_trn+nCV_tst: print "Error: training and test don't sum up"

# Within sss in iloc coordinates
Ix_trn = np.zeros((nCV_trn, nCV))
Ix_tst = np.zeros((nCV_tst, nCV))
for i, (train_index, test_index) in enumerate(sss):
    Ix_trn[:,i] = y_trn.index[train_index]
    Ix_tst[:,i] = y_trn.index[test_index]
    Ix_trn[:,i] = y_trn.index[train_index]
    Ix_tst[:,i] = y_trn.index[test_index]
Ix_trn = Ix_trn.astype(int)
Ix_tst = Ix_tst.astype(int)

# Select CV train and CV test sets
i0 = 0 # e.g. select first fold
X_CV_trn, y_CV_trn = X_trn.ix[Ix_trn[:,i0]], y_trn.ix[Ix_trn[:,i0]]
X_CV_tst, y_CV_tst = X_trn.ix[Ix_tst[:,i0]], y_trn.ix[Ix_tst[:,i0]]


## This would suggest that if you use:
#review_count
#attributes.Drive-Thru
#attributes.Takes Reservations
#attributes.Parking.street
#hours.Monday.close
#attributes.Has TV
#stars
#You might be able to tell the country apart

from sklearn.metrics import confusion_matrix
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)
#pred_varbl = ['review_count', 'stars', 'attributes.Takes Reservations', \
#'attributes.Parking.street', 'hours.Monday.close', \
#'attributes.Has TV']
# With this I get 2 predicted countries, max_depth=3
pred_varbl = ['review_count', 'stars', 'attributes.Takes Reservations', \
'attributes.Parking.street', \
'hours.Sunday.open', 'hours.Sunday.close', \
'hours.Monday.close', \
'hours.Thursday.close', \
'hours.Friday.open', \
'hours.Friday.close', \
'hours.Saturday.open', \
'hours.Saturday.close', \
'attributes.Has TV', 'attributes.Dogs Allowed', \
'attributes.Drive-Thru', 'attributes.Alcohol', \
'attributes.Waiter Service', \
'attributes.Good For.lunch', \
'attributes.Dietary Restrictions.dairy-free', \
'attributes.Dietary Restrictions.gluten-free']
# With this I get 3 predicted countries, max_depth=3

## There seem to still be NaN's
#any(X_CV_trn.ix[:,pred_varbl].isnull())
#
#for nm, col in X_CV_trn.iteritems():
#    print nm, any(X_CV_trn.ix[:,nm].isnull())
#    if any(X_CV_trn.ix[:,nm].isnull())==True:
#        print 'Hellooooooooooo????'
## Ehhh?????
#for i in range(len(pred_varbl)):
#    print i, any(X_CV_trn.ix[:,pred_varbl[i]].isnull())
## Mispelled Thursday ...

#for i in range(len(pred_varbl)):
#    print i, any(X_CV_trn.ix[:,pred_varbl[i]].isnull())
## Mispelled Thursday ...

clf.fit(X_CV_trn.ix[:,pred_varbl], y_CV_trn)
y_CV_pred = clf.predict(X_CV_tst.ix[:,pred_varbl])
Z = pd.DataFrame([y_CV_pred, y_CV_tst], index=['pred', 'act']).T
M = confusion_matrix(y_CV_tst, y_CV_pred)
print M

# Something quite wrong here, the confusion matrix has a lot of zeros AND
# it does not change if I vary the set of predictive variables

# RE-START FROM HERE

y_CV_tst.value_counts()
# Ok, so we have all representative states

y_CV_pred.value_counts() # not good for np array
y_CV_pred = pd.Series(y_CV_pred, index=y_CV_tst.index)
y_CV_pred.value_counts() # now good
# Well it always predicts state=3 ...

y_CV_trn.value_counts()
# Ok, also distributed

# So it seems the fit is really poor, hoa about all?
clf.fit(X_CV_trn, y_CV_trn)
y_CV_pred = clf.predict(X_CV_tst)
Z = pd.DataFrame([y_CV_pred, y_CV_tst], index=['pred', 'act']).T
M = confusion_matrix(y_CV_tst, y_CV_pred)
print M
# Wow, that seems a perfect score! On test set! Wow
# Well, I thing geographical location will do the job!

# Ok, so results ok with max depth=5

# 1) convert confusion matrix in F-score, I think to remember
# 2) Use CV dof to fine-tune
# Experiment with alternative classification models



# BIG NEW TOPIC: NPL








## This was my stuff ...
#df_eg = pd.DataFrame({'e_gain': egs}, index=ixsDT)
#df_DT = pd.DataFrame({'DT': zs}, index=ixsDT)
#df = df_eg.join(df_DT)
#
#
#ind = np.arange(df.shape[0])
#fig = pl.figure(figsize=(9, 4))
#ax = fig.add_subplot(111)
#ax.bar(ind,df.e_gain, width=0.25)
#ax.bar(ind+0.5,df.DT, width=0.25, color='g')
#pl.xticks(ind, ixs_eg, rotation=90)
#
#
#
#
#
#
### The following is for bucketing, I will skip for now
##data_aug = df_res.copy()
##for nm, sp in df_types['support'].iteritems():
##    print nm, sp
##    if sp=='continuous':
##        print df_types.loc[nm]
##        # Create buckets
##        nB = 5
##        col_num = create_buckets(data_aug[nm], nB)
##        data_aug = pd.concat([data_aug, col_num], axis=1)
##        out_tmp = cat_series2num(data_aug[col_num.name]) # convert string to numeric level
##        col_num = out_tmp['x_num']
##        col_num.name = nm + '_num'
##        data_aug = pd.concat([data_aug, col_num], axis=1)
##    else:
##        # Create the couple of (bucket txt value, bucket numerical value)
##        # Also create a dictionary of that
##        if df_types.ix[nm, 'type'] in (np.int64, np.float64):
##            col_txt = nm + '_' + data_aug[nm].astype(str) # convert numeric level to string
##            col_txt.name = nm + '_lev'
##            data_aug = pd.concat([data_aug, col_txt], axis=1)
##        elif df_types.ix[nm, 'type']==np.object:
##            out_tmp = cat_series2num(data_aug[nm]) # convert string to numeric level
##            col_num = out_tmp['x_num']
##            col_num.name = nm + '_num'
##            data_aug = pd.concat([data_aug, col_num], axis=1)
#
#
## Split data
#y = df_res['stars']
#X = df_res.drop('stars', axis=1)
#n_all = len(X)
#n_tst = int(n_all*0.1)
#n_trn = n_all - n_tst
#
#
## Sampling with stratified sampling
#from sklearn.cross_validation import StratifiedShuffleSplit
#
#strf = StratifiedShuffleSplit(y, n_iter=1, test_size=n_tst, random_state=0)
#for ix_trn, ix_tst in strf: _ = False
#X_trn, y_trn = X.ix[ix_trn, :], y[ix_trn]
#X_tst, y_tst = X.ix[ix_tst, :], y[ix_tst]
## ixs = np.concatenate((ix_trn, ix_tst))
#
## Look at prj2_students_interv_20160506.py
#
## Review this part!
#
## !!!!!!!!!!!!!!!!!!!
#
### Sets for CV
##pctCV_tst = 0.15 # CV test
##nCV_tst = int(float(n_trn)*pctCV_tst)
##nCV_trn = n_trn - nCV_tst
##if nCV_trn + nCV_tst!=n_trn: print "Error: training and test don't sum up"
##
##nCV = 3
##sss = StratifiedShuffleSplit(y_trn, nCV, test_size=nCV_tst, random_state=0)
##Ix_trn = np.zeros((nCV_trn, nCV))
##Ix_tst = np.zeros((nCV_tst, nCV))
###i = 0
##for i, (train_index, test_index) in enumerate(sss):
##    Ix_trn[:,i] = y_trn.index[train_index]
##    Ix_tst[:,i] = y_trn.index[test_index]
###    i += 1
##Ix_trn = Ix_trn.astype(int)
##Ix_tst = Ix_tst.astype(int)
##
### Select CV train and CV test sets
##i0 = 0 # e.g. select first fold
##X_CV_trn, y_CV_trn = X_trn.loc[Ix_trn[:,i0]], y_trn[Ix_trn[:,i0]]
##X_CV_tst, y_CV_tst = X_trn.loc[Ix_tst[:,i0]], y_trn[Ix_tst[:,i0]]
#
#
#
## Important features
#from rg_toolbox_ml import train_clf, pred_clf, run_clf
#
## Decision Tree
#from sklearn.tree import DecisionTreeClassifier
#clf_DT_ = DecisionTreeClassifier(max_depth=3)
##out_trn_DT_ = train_clf(clf_DT_, X_CV_trn, y_CV_trn);
##out_prd_DT_ = pred_clf(clf_DT_, X_CV_trn, y_CV_trn)
#out_trn_DT_ = train_clf(clf_DT_, X_trn, y_trn)
#out_prd_DT_ = pred_clf(clf_DT_, X_trn, y_trn)
#print "Prediction with %s F1 = %1.4f" % (clf_DT_.__class__.__name__ ,out_prd_DT_['score'])
#
#
##
##X_tmp_aug, y_tmp = X_org_aug.reindex(ixs), y_org.reindex(ixs)
##
##X_train, y_train = X_tmp_aug.ix[ixs[:num_train],:], y_tmp[ixs[:num_train]]
##X_test, y_test = X_tmp_aug.ix[ixs[num_train:],:], y_tmp[ixs[num_train:]]
##
##print "Training set: {} samples".format(X_train.shape[0])
##print "Test set: {} samples".format(X_test.shape[0])
### Note: If you need a validation set, extract it from within training data
#
#
#
## Determinants of rating per country
#
