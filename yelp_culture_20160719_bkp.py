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

# Isn't this simpler? Well, it's the same thing just with more labels
df_num['stars'].value_counts()


#f, ax = plt.subplots(figsize=(10,4))
#n = 35
#x = [i-0.25 for i in range(n)]
#ax.bar(x, df_grp['cnt'].head(n), width=0.5);
#ax.set_title('Count per (Joint) Category')
#ax.set_xticks(range(n));
#ax.set_xticklabels(df_grp['cat'].iloc[:n], rotation='vertical');
#ax.set_xlim((-0.25, n));
#if sav: plt.savefig(dnm + '\\' + 'Business_cat.png', bbox_inches='tight')
#
## df_grp['cnt'].where(df_grp['cnt']==1) # not really
#ix_1 = np.where(df_grp['cnt']==1) # type(ix) # tuple
#ix_1bp = np.where(df_grp['cnt']==int(0.0001*nR[nm0]))
#ix_1[0][0]
#ix_1bp[0][0]
## Actually you want say 99% of the corpus being described
#ix_99 = np.where(df_grp['cnt'].cumsum()>=0.99*nR[nm0])
#ix_99[0][0]
## You have to go pretty far into the tail to get 99% coverage


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


# Clean data
cityU = list(df_res['city'].unique())
cityU.sort()
# Needs clean-up, but is it necessary?
stateU = list(df_res['state'].unique())
stateU.sort()

df_res.ix[::10,['city', 'state']]


# Augment data
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


# Prepare the data

# Jeez!! You need to convert to numerical values!!!

import sys; sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND')
from rg_toolbox_data import data_types, cat_series2num, str2timestamp

df_supp_types = data_types(df_res)  # if variable has continuous or categorical (discrete) support; I guess categorical could be ordered or not
# Nope: detection of time variable sucks!
# Ok, improved

df_num = df_res.copy()
for nm in df_num.columns:
    print "Column '{}' is of '{}' support".format(nm, df_supp_types.ix[nm, 'support'])
    if not(all(df_num[nm].isnull())):
        if df_supp_types.ix[nm, 'support']=='discrete' and df_supp_types.ix[nm, 'type']=='string':
            out_tmp = cat_series2num(df_res[nm]) # convert string to numeric level
            col_num = out_tmp['x_num']
            col_num.name = nm + '_num'
            df_num[nm] = col_num
        elif (df_supp_types.ix[nm, 'subtype']=='int') and (type(df_num[nm].iloc[0])==np.str):
            df_num[nm] = df[nm].astype(float)
            df_num[nm][df_num[nm].isnull()] = df_num[nm].mean()
            df_num[nm] = df_num[nm].astype(int)
        elif (df_supp_types.ix[nm, 'subtype']=='float') and (type(df_num[nm].iloc[0])==np.str):
            df_num[nm] = df[nm].astype(float)
        elif df_supp_types.ix[nm, 'type']=='time':
            df_num[nm] = pd.to_datetime(df_num[nm])
    #    elif df_supp_types.ix[nm, 'support']=='continuous' and df_supp_types.ix[nm, 'type']=='time':
    #        df_num[nm].to_timestamp()
    #    elif df_supp_types.ix[nm, 'support']=='continuous' and df_supp_types.ix[nm, 'type']=='time':
    #        # Convert times into timestamp
    ##        z = df_num[nm].apply('totimestamp')
    ##        totimestamp(dateutil.parser.parse((df_num[nm].iloc[0])))
    ##        str2timestamp(df_num[nm].iloc[0]) # this works
    ##        z = df_num[nm].apply(str2timestamp) # this doesn't
    #    
    #        # -RG 7/18/2018: this is too slow I will remove it for now
    #        for i in range(len(df_num[nm])):
    #            print 'Iteration {}/{}'.format(i, len(df_num))
    #            df_num[nm].iloc[i] = str2timestamp(df_num[nm].iloc[i])
df_num.info()

# Awesome, then, let's see if I can pass it on ML routines ...

# Handle exceptions
for nm in ['state', 'city']:
    out_tmp = cat_series2num(df_res[nm]) # convert string to numeric level
    col_num = out_tmp['x_num']
    col_num.name = nm + '_num'
    df_num[nm] = col_num

df_num.info()
df_num.head()


## Get rid of all columns of type=time
#for nm in df_num.columns:
#    if (all(df_num[nm].isnull())) or df_supp_types.ix[nm, 'type']=='time':
#        df_num.drop(nm, axis=1, inplace=True)

# Get rid of other columns
for nm in ['neighborhoods', 'full_address', 'name', 'categories', 'business_id']:
    df_num.drop(nm, axis=1, inplace=True)

df_num.info()
df_num.head()


## Jeez, there is still crap
#df_num['neighborhoods'] # these are lists
#df_num['state'] # not sure why this is an object ...
#
#for i in range(len(df_num)):
#    if type(df_num['state'].iloc[0])!=np.str:
#        print i

# Ok, now all numeric representations


# Decide output variable: stars
y = df_num['stars']
X = df_num.drop('stars', axis=1)
nO = len(X)
nTst = int(nO*0.15)
nTrn = nO - nTst


#len(df_num[df_num['stars'].isnull(),'stars'])

#ix_na = []
#for i in range(len(df_num)):
#    if df_num['stars'].iloc[i].isnull()==True:
#        ix_na.append(i)

# Split data set

# P-Reshuffling: the first shuffle is just to extract the test set, not for CV
from sklearn.cross_validation import StratifiedShuffleSplit
sss0 = StratifiedShuffleSplit(y, 1, test_size=nTst, random_state=0)
for ix_trn, ix_tst in sss0: _ = False
X_trn, y_trn = X.ix[ix_trn], y[ix_trn]
X_tst, y_tst = X.ix[ix_tst], y[ix_tst]
any(np.isnan(y))
any(y_trn.isnull())
# HOW THE HELL IS THIS POSSIBLE????



# Check distribution of y variables
grp = pd.value_counts(df_num['stars'])
df_grp = pd.DataFrame({'cat': grp.index, 'cnt': grp.values})
print df_grp

df_num['stars'].value_counts()
df_num.ix[ix_trn, 'stars'].value_counts()
df_num.ix[ix_tst, 'stars'].value_counts()
# So far so good, it seems
# Ah, but then if I do 3 CV it might be tricky
# Although it should only be for the training sample, so that should be ok, no?
# Still VERY strange!
# Check by hand

##y_tmp = y_trn.where(y_trn==1.0) # doesn't work
#len(y_trn[y_trn==1.0]) # = 0????
#len(y_trn[y_trn==1.5]) # = 0????
#
y_trn.value_counts()
any(y_trn.isnull())
# Damn!!!!!


ix_tmp = ix_trn.copy()
ix_tmp.sort()
# Note: ix_trn is (correctly) in 'index' units not in 'iloc'

# Sets for CV
pctCV_tst = 0.15 # CV test
nCV_tst = int(nTrn*pctCV_tst)
nCV_trn = nTrn - nCV_tst
if nCV_trn+nCV_tst!=nTrn: print "Error: training and test don't sum up"

nCV = 3
sss = StratifiedShuffleSplit(y_trn, nCV, test_size=nCV_tst, random_state=1)
Ix_trn = np.zeros((nCV_trn, nCV))
Ix_tst = np.zeros((nCV_tst, nCV))
for i, (train_index, test_index) in enumerate(sss):
    print i
#    print train_index
#    print test_index
#    print train_index[:5]
#    print y_trn[train_index[:5]]
#    Ix_trn[:,i] = y_trn.index[train_index]
#    Ix_tst[:,i] = y_trn.index[test_index]
#    Ix_trn[:,i] = train_index]
#    Ix_tst[:,i] = y_trn.index[test_index]
Ix_trn = Ix_trn.astype(int)
Ix_tst = Ix_tst.astype(int)


# Select CV train and CV test sets
i0 = 0 # e.g. select first fold
X_CV_trn, y_CV_trn = X_trn.loc[Ix_trn[:,i0]], y_trn[Ix_trn[:,i0]]
X_CV_tst, y_CV_tst = X_trn.loc[Ix_tst[:,i0]], y_trn[Ix_tst[:,i0]]



# Using a tree for feature importance
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf.fit(X_train, y_train)
z = Series(clf.feature_importances_, X_train.columns)
zs = z.order(ascending=False)
ixsDT = zs.index

df_eg = DataFrame({'e_gain': egs}, index=ixs_eg)
df_DT = DataFrame({'DT': zs}, index=ixsDT)
df = df_eg.join(df_DT)


ind = np.arange(df.shape[0])
fig = pl.figure(figsize=(9, 4))
ax = fig.add_subplot(111)
ax.bar(ind,df.e_gain, width=0.25)
ax.bar(ind+0.5,df.DT, width=0.25, color='g')
pl.xticks(ind, ixs_eg, rotation=90)






## The following is for bucketing, I will skip for now
#data_aug = df_res.copy()
#for nm, sp in df_types['support'].iteritems():
#    print nm, sp
#    if sp=='continuous':
#        print df_types.loc[nm]
#        # Create buckets
#        nB = 5
#        col_num = create_buckets(data_aug[nm], nB)
#        data_aug = pd.concat([data_aug, col_num], axis=1)
#        out_tmp = cat_series2num(data_aug[col_num.name]) # convert string to numeric level
#        col_num = out_tmp['x_num']
#        col_num.name = nm + '_num'
#        data_aug = pd.concat([data_aug, col_num], axis=1)
#    else:
#        # Create the couple of (bucket txt value, bucket numerical value)
#        # Also create a dictionary of that
#        if df_types.ix[nm, 'type'] in (np.int64, np.float64):
#            col_txt = nm + '_' + data_aug[nm].astype(str) # convert numeric level to string
#            col_txt.name = nm + '_lev'
#            data_aug = pd.concat([data_aug, col_txt], axis=1)
#        elif df_types.ix[nm, 'type']==np.object:
#            out_tmp = cat_series2num(data_aug[nm]) # convert string to numeric level
#            col_num = out_tmp['x_num']
#            col_num.name = nm + '_num'
#            data_aug = pd.concat([data_aug, col_num], axis=1)


# Split data
y = df_res['stars']
X = df_res.drop('stars', axis=1)
n_all = len(X)
n_tst = int(n_all*0.1)
n_trn = n_all - n_tst


# Sampling with stratified sampling
from sklearn.cross_validation import StratifiedShuffleSplit

strf = StratifiedShuffleSplit(y, n_iter=1, test_size=n_tst, random_state=0)
for ix_trn, ix_tst in strf: _ = False
X_trn, y_trn = X.ix[ix_trn, :], y[ix_trn]
X_tst, y_tst = X.ix[ix_tst, :], y[ix_tst]
# ixs = np.concatenate((ix_trn, ix_tst))

# Look at prj2_students_interv_20160506.py

# Review this part!

# !!!!!!!!!!!!!!!!!!!

## Sets for CV
#pctCV_tst = 0.15 # CV test
#nCV_tst = int(float(n_trn)*pctCV_tst)
#nCV_trn = n_trn - nCV_tst
#if nCV_trn + nCV_tst!=n_trn: print "Error: training and test don't sum up"
#
#nCV = 3
#sss = StratifiedShuffleSplit(y_trn, nCV, test_size=nCV_tst, random_state=0)
#Ix_trn = np.zeros((nCV_trn, nCV))
#Ix_tst = np.zeros((nCV_tst, nCV))
##i = 0
#for i, (train_index, test_index) in enumerate(sss):
#    Ix_trn[:,i] = y_trn.index[train_index]
#    Ix_tst[:,i] = y_trn.index[test_index]
##    i += 1
#Ix_trn = Ix_trn.astype(int)
#Ix_tst = Ix_tst.astype(int)
#
## Select CV train and CV test sets
#i0 = 0 # e.g. select first fold
#X_CV_trn, y_CV_trn = X_trn.loc[Ix_trn[:,i0]], y_trn[Ix_trn[:,i0]]
#X_CV_tst, y_CV_tst = X_trn.loc[Ix_tst[:,i0]], y_trn[Ix_tst[:,i0]]



# Important features
from rg_toolbox_ml import train_clf, pred_clf, run_clf

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf_DT_ = DecisionTreeClassifier(max_depth=3)
#out_trn_DT_ = train_clf(clf_DT_, X_CV_trn, y_CV_trn);
#out_prd_DT_ = pred_clf(clf_DT_, X_CV_trn, y_CV_trn)
out_trn_DT_ = train_clf(clf_DT_, X_trn, y_trn)
out_prd_DT_ = pred_clf(clf_DT_, X_trn, y_trn)
print "Prediction with %s F1 = %1.4f" % (clf_DT_.__class__.__name__ ,out_prd_DT_['score'])


#
#X_tmp_aug, y_tmp = X_org_aug.reindex(ixs), y_org.reindex(ixs)
#
#X_train, y_train = X_tmp_aug.ix[ixs[:num_train],:], y_tmp[ixs[:num_train]]
#X_test, y_test = X_tmp_aug.ix[ixs[num_train:],:], y_tmp[ixs[num_train:]]
#
#print "Training set: {} samples".format(X_train.shape[0])
#print "Test set: {} samples".format(X_test.shape[0])
## Note: If you need a validation set, extract it from within training data



# Determinants of rating per country

