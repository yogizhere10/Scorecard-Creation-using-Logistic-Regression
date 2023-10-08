#!/usr/bin/env python
# coding: utf-8

# # Scorecard creation using Inbuilt library

# In[1]:


import pandas as pd
import numpy as np

pd.options.display.max_columns=None

from IPython.display import display

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")


# In[2]:


import toad


# In[3]:


df = pd.read_csv("loan_data_2007_2014.csv")


# In[4]:


df.drop(["Unnamed: 0",'id','member_id' ,'url','desc'], axis = 1, inplace = True)
df.head(2)


# In[5]:


df.shape


# In[6]:


df = df.loc[: 50000,:]
df.shape


# In[7]:


toad.detect(df)[:30]


# In[8]:


# 0 means bad/default and 1 means good/non-default
df['good_bad'] = np.where(
    df['loan_status'].isin(
        ['Charged Off','Default','Late (31-120 days)','Does not meet the credit policy. Status:Charged Off']),
        1,0)


# In[9]:


df.drop(['loan_status'], axis = 1, inplace = True)


# In[10]:


toad.quality(df, 'good_bad', iv_only = True)[:20]


# In[11]:


df_select, dropped = toad.selection.select(df, target = 'good_bad', empty = 0.55, iv = 0.05, corr = 0.75, return_drop = True)


# In[12]:


dropped


# In[13]:


df_select.drop(['emp_title', 'title', 'zip_code', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d'], axis = 1, inplace = True)
df_select.shape


# In[14]:


df['last_pymnt_d'].nunique()


# In[15]:


c = toad.transform.Combiner()


# In[16]:


c.fit(df_select, y = 'good_bad', method = 'chi', min_samples = 0.05)


# In[17]:


c.export()


# In[18]:


from toad.plot import bin_plot


# In[19]:


bin_plot(c.transform(df_select[['last_pymnt_amnt','good_bad']], labels=True), x='last_pymnt_amnt', target='good_bad')


# In[20]:


bin_plot(c.transform(df_select[['recoveries','good_bad']], labels=True), x='recoveries', target='good_bad')


# In[21]:


bin_plot(c.transform(df_select[['total_rec_late_fee','good_bad']], labels=True), x='total_rec_late_fee', target='good_bad')


# In[22]:


bin_plot(c.transform(df_select[['total_rec_int','good_bad']], labels=True), x='total_rec_int', target='good_bad')


# In[23]:


bin_plot(c.transform(df_select[['total_rec_prncp','good_bad']], labels=True), x='total_rec_prncp', target='good_bad')


# In[24]:


bin_plot(c.transform(df_select[['out_prncp_inv','good_bad']], labels=True), x='out_prncp_inv', target='good_bad')


# In[25]:


bin_plot(c.transform(df_select[['revol_util','good_bad']], labels=True), x='revol_util', target='good_bad')


# In[26]:


bin_plot(c.transform(df_select[['inq_last_6mths','good_bad']], labels=True), x='inq_last_6mths', target='good_bad')


# In[27]:


bin_plot(c.transform(df_select[['purpose','good_bad']], labels=True), x='purpose', target='good_bad')


# In[28]:


bin_plot(c.transform(df_select[['issue_d','good_bad']], labels=True), x='issue_d', target='good_bad')


# In[29]:


bin_plot(c.transform(df_select[['annual_inc','good_bad']], labels=True), x='annual_inc', target='good_bad')


# In[30]:


bin_plot(c.transform(df_select[['sub_grade','good_bad']], labels=True), x='sub_grade', target='good_bad')


# In[31]:


bin_plot(c.transform(df_select[['grade','good_bad']], labels=True), x='grade', target='good_bad')


# In[32]:


bin_plot(c.transform(df_select[['int_rate','good_bad']], labels=True), x='int_rate', target='good_bad')


# In[33]:


transer = toad.transform.WOETransformer()


# In[34]:


df_woe = transer.fit_transform(c.transform(df_select), df_select['good_bad'], exclude=['good_bad'])


# In[35]:


df_woe.shape


# In[36]:


df_woe.head()


# In[37]:


final_data = toad.selection.stepwise(df_woe,target = 'good_bad', estimator='ols', direction = 'both', criterion = 'aic')


# In[38]:


col = list(final_data.drop(['good_bad'],axis=1).columns)


# In[39]:


col


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


xtrain, xtest, ytrain, ytest = train_test_split(final_data, df_woe['good_bad'], test_size = 0.2, random_state = 0)
xtrain.shape, xtest.shape, ytrain.shape, ytest.shape


# In[42]:


from sklearn.linear_model import LogisticRegression


# In[43]:


lr = LogisticRegression()
lr.fit(xtrain, ytrain)


# In[44]:


pred_test = lr.predict_proba(xtest)[:,1]


# In[45]:


pred_test


# In[46]:


ytest


# In[47]:


ytest.reset_index(drop = True, inplace = True)
ytest.head()


# In[48]:


df_actual_predicted_probs = pd.concat([ytest, pd.DataFrame(pred_test)], axis = 1)


# In[49]:


df_actual_predicted_probs.columns = ['loan_data_targets_test', 'y_hat_test_proba']


# In[50]:


df_actual_predicted_probs.head()


# In[51]:


tr = 0.85
# We create a new column with an indicator,
# where every observation that has predicted probability greater than the threshold has a value of 1,
# and every observation that has predicted probability lower than the threshold has a value of 0.
df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)


# In[52]:


pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted'])


# In[53]:


pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]


# In[54]:


(pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[0, 0] 


# In[55]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[56]:


roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])


# In[57]:


fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])


# In[58]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[59]:


plt.plot(fpr, tpr)
# We plot the false positive rate along the x-axis and the true positive rate along the y-axis,
# thus plotting the ROC curve.
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
# We plot a seconary diagonal line, with dashed line style and black color.
plt.xlabel('False positive rate')
# We name the x-axis "False positive rate".
plt.ylabel('True positive rate')
# We name the x-axis "True positive rate".
plt.title('ROC curve')
# We name the graph "ROC curve".


# In[60]:


AUROC = roc_auc_score(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC


# In[61]:


df_actual_predicted_probs = df_actual_predicted_probs.reset_index()


# In[62]:


df_actual_predicted_probs['Cumulative_N_Population'] = df_actual_predicted_probs.index + 1
df_actual_predicted_probs['Cumulative_N_Good'] = df_actual_predicted_probs['loan_data_targets_test'].cumsum()
df_actual_predicted_probs['Cumulative_N_Bad'] = df_actual_predicted_probs['Cumulative_N_Population'] - df_actual_predicted_probs['loan_data_targets_test'].cumsum()
df_actual_predicted_probs.columns


# In[63]:


df_actual_predicted_probs['Cumulative_Perc_Population'] = df_actual_predicted_probs['Cumulative_N_Population'] / (df_actual_predicted_probs.shape[0])
df_actual_predicted_probs['Cumulative_Perc_Good'] = df_actual_predicted_probs['Cumulative_N_Good'] / df_actual_predicted_probs['loan_data_targets_test'].sum()
df_actual_predicted_probs['Cumulative_Perc_Bad'] = df_actual_predicted_probs['Cumulative_N_Bad'] / (df_actual_predicted_probs.shape[0] - df_actual_predicted_probs['loan_data_targets_test'].sum())
df_actual_predicted_probs.head()


# In[71]:


# Plot Gini
#plt.plot(df_actual_predicted_probs['Cumulative_Perc_Population'], df_actual_predicted_probs['Cumulative_Perc_Bad'],linestyle = '-', color = 'r')
# We plot the cumulative percentage of all along the x-axis and the cumulative percentage 'good' along the y-axis,
# thus plotting the Gini curve.
plt.plot(df_actual_predicted_probs['Cumulative_Perc_Population'], df_actual_predicted_probs['Cumulative_Perc_Population'], linestyle = '--', color = 'k')
# We plot a seconary diagonal line, with dashed line style and black color.
plt.xlabel('Cumulative % Population')
# We name the x-axis "Cumulative % Population".
plt.ylabel('Cumulative % Bad')
# We name the y-axis "Cumulative % Bad".
plt.title('Gini')
# We name the graph "Gini".


# In[65]:


Gini = AUROC * 2 - 1
# Here we calculate Gini from AUROC.
Gini


# In[66]:


KS = max(df_actual_predicted_probs['Cumulative_Perc_Bad'] - df_actual_predicted_probs['Cumulative_Perc_Good'])
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS


# In[67]:


from toad.metrics import KS, AUC


# In[68]:


print('test KS',KS(pred_test, ytest))
print('test AUC',AUC(pred_test, ytest))


# In[69]:


# Group the predicted scores in bins with same number of samples in each (i.e. "quantile" binning)
toad.metrics.KS_bucket(pred_test, ytest, bucket=10, method = 'quantile')


# In[76]:


ytrain


# In[70]:


card = toad.ScoreCard(
    combiner = c,
    transer = transer,
    #class_weight = 'balanced',
    #C=0.1,
    #base_score = 600,
    #base_odds = 35 ,
    #pdo = 60,
    #rate = 2
)


# In[77]:


card.fit(final_data[col], final_data['good_bad'])


# In[78]:


# Output standard scorecard
card.export()


# In[ ]:




