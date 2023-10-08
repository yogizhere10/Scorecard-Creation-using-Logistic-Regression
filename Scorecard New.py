#!/usr/bin/env python
# coding: utf-8

# # Scorecard Development

# In[1]:


#Import Required Libraries
import numpy as np
import pandas as pd


# In[2]:


#Read Input file
df = pd.read_csv("loan_data_2015.csv")


# In[3]:


#Import Important Libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency


# In[4]:


pd.options.display.max_columns = None


# In[5]:


df.head()


# In[6]:


df.columns


# In[7]:


df.info()


# # Missing value Treatment

# In[8]:


nulls = df.isnull().mean()


# In[9]:


#Delete All columns with Nulls Greater than 80%
nulls_del_lst = nulls[nulls > 0.8].index.tolist()


# In[10]:


df.drop(nulls_del_lst, axis = 1, inplace = True)


# In[11]:


df.shape


# In[12]:


[df.emp_title.value_counts() > 1]


# In[13]:


# Drop all Not required columns
df.drop(columns = ['id', 'member_id', 'sub_grade', 'emp_title', 'url', 'title', 'zip_code', 'next_pymnt_d',
            'recoveries', 'collection_recovery_fee', 'total_rec_prncp', 'total_rec_late_fee'], inplace = True)


# In[14]:


df.head()


# In[15]:


df['loan_status'].value_counts(normalize = True)


# In[16]:


# Creating Target variable, assigning 1 for Good and 0 for bad for coding and logic
df['good_bad'] = np.where(df['loan_status'].isin(['Charged Off', 'Default', 'Late (31-120 days)',
                                    'Does not meet the credit policy. Status:Charged Off']),0,1)


# In[17]:


df['good_bad'].value_counts(normalize = True)


# In[18]:


df.drop('loan_status', axis = 1, inplace = True)


# In[19]:


df['emp_length'].value_counts(), df['emp_length'].unique()


# ## Data Trasnformation as required

# In[20]:


# A Function to convert Emp_Length variable into required format and convert to Numeric

def emp_convert(ds, col):
    ds[col] = ds[col].str.replace("< 1 year", str(0))
    ds[col] = ds[col].str.replace("\+ years", "")
    ds[col] = ds[col].str.replace(" years", "")
    ds[col] = ds[col].str.replace(" year", "")
    
    ds[col] = pd.to_numeric(ds[col])


# In[21]:


emp_convert(df, "emp_length")


# In[22]:


df['emp_length'].value_counts()


# In[23]:


# A Function to transfer all dates into Difference of Month between today and date as mentioned in respective column
def date_convert(ds, col):
    today = pd.to_datetime("2023-07-01")
    ds[col] = pd.to_datetime(ds[col], format = "%b-%y")
    ds['mths_since_' + col] = round(pd.to_numeric((today - ds[col]) / np.timedelta64(1,"M")))
    ds['mths_since_' + col] = np.where(ds['mths_since_' + col] < 0, ds['mths_since_' + col].max(), 
                                       ds['mths_since_' + col])
    ds.drop(col, axis = 1, inplace = True)


# In[24]:


date_convert(df, 'earliest_cr_line')


# In[25]:


df['mths_since_earliest_cr_line'].describe()


# In[26]:


date_convert(df, 'issue_d')


# In[27]:


date_convert(df, 'last_pymnt_d')


# In[28]:


date_convert(df, 'last_credit_pull_d')


# In[29]:


print(df['mths_since_earliest_cr_line'].describe())
print(df['mths_since_issue_d'].describe())
print(df['mths_since_last_pymnt_d'].describe())
print(df['mths_since_last_credit_pull_d'].describe())


# In[30]:


df['term'].value_counts()


# In[31]:


df['term'] = df['term'].str.replace(" months", "")
df['term'] = pd.to_numeric(df['term'])


# In[32]:


x = df.drop('good_bad', axis = 1)
y = df['good_bad']


# In[33]:


purpose_grp = df['purpose'].value_counts(ascending = True).index.tolist()[:3]


# In[34]:


# Data tranformation for Purpose variable
df['purpose'] = np.where(df['purpose'].isin(purpose_grp), "other", df['purpose'])


# In[35]:


df['initial_list_status'].value_counts()


# In[36]:


#df.drop('initial_list_status', axis = 1, inplace = True)


# ## Bifurcating Numeric and categorical Variables for Further Processing

# In[37]:


# Bifurcating categorical and Numeric variables

cat = x.select_dtypes(include = 'object')
num = x.select_dtypes(include = 'number')


# In[38]:


cat.shape, num.shape


# ## Variables selection for categorical variable Using Chi_Square

# In[39]:


# This code will find relevance of Categorical variables and help us to discard variable based on Pvalue


# In[40]:


chi2_res = {}

for col in cat.columns:
    chi, p, dof, ex = chi2_contingency(pd.crosstab(y, cat[col]))
    chi2_res.setdefault("Feature", []).append(col)
    chi2_res.setdefault("p-value",[]).append(round(p,6))
    chi2_res.setdefault("Chi_val", []).append(chi)


# In[41]:


chi2_df = pd.DataFrame(data = chi2_res)


# In[42]:


chi2_df


# In[43]:


chi2_df.sort_values(by = ['p-value'], ascending = True, 
                    ignore_index = True, inplace = True)


# In[44]:


# Not considering last variable "application_type"
chi2_df = chi2_df[:-1]


# In[45]:


cat_lst = chi2_df.Feature.values.tolist()


# In[46]:


cat = cat[cat_lst]


# In[47]:


#Dropping "addr_state", "pymnt_plan"
cat.drop(['addr_state', 'pymnt_plan'], axis = 1, inplace = True)


# In[48]:


cat.shape, num.shape


# In[49]:


num.isnull().mean()


# In[50]:


df['mths_since_last_major_derog'].value_counts(normalize = True)


# In[51]:


df['mths_since_last_delinq'].value_counts(normalize = True).index.tolist()[0]


# In[52]:


df['emp_length'].value_counts(normalize = True).index.tolist()[0]


# In[53]:


df['emp_length'].fillna(df['emp_length'].value_counts(normalize = True).index.tolist()[0], inplace = True)


# In[54]:


num['emp_length'].value_counts(normalize = True)


# In[55]:


num['emp_length'].fillna(num['emp_length'].value_counts(normalize = True).index.tolist()[0], inplace = True)


# In[56]:


num['mths_since_last_delinq'].fillna(num['mths_since_last_delinq'].value_counts(normalize = True).index.tolist()[0], inplace = True)


# In[57]:


num['revol_util'].fillna(num['revol_util'].mean(), inplace = True)


# In[58]:


num['mths_since_last_delinq'].fillna(0, inplace = True)


# In[59]:


num.drop(['mths_since_last_major_derog'], axis = 1, inplace = True)


# In[60]:


num['mths_since_last_pymnt_d'].fillna(num['mths_since_last_pymnt_d'].mean(), inplace = True)
num['mths_since_last_credit_pull_d'].fillna(num['mths_since_last_credit_pull_d'].mean(), inplace = True)


# In[61]:


cat.head()


# In[62]:


cat.isnull().sum()


# ## Variable selection for Numeric Variable 

# In[63]:


fstats = f_classif(num, y)[0]
pval = f_classif(num, y)[1]


# In[64]:


fstats, pval


# In[65]:


anova_df = pd.DataFrame(data = {'Num_feature': num.columns.values,
                                'F-Score' : fstats,
                                'p_value' : pval})


# In[66]:


anova_df.sort_values(by = 'F-Score', ascending = True, 
                    ignore_index = True, inplace = True)


# In[67]:


anova_df.dropna(inplace=True)


# In[68]:


anova_df = anova_df.loc[5:,:]


# In[69]:


anova_df.reset_index(inplace = True, drop = True)


# In[70]:


anova_df


# In[71]:


top_num_col = anova_df['Num_feature'].values.tolist()


# In[72]:


corrmat = num[top_num_col].corr()


# In[73]:


mask = np.triu(np.ones_like(corrmat, dtype = np.bool))


# In[74]:


f, ax = plt.subplots(figsize=(20,20))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corrmat, cmap= cmap, mask= mask, vmax=1, center=0.5,
            square=True, linewidths=.5, cbar_kws={"shrink": .6},annot=True)

plt.title("Pearson correlation", fontsize =10)


# In[75]:


num.corr().columns


# ## Function to Remove Multi Collinearity

# In[76]:


def correlation(data, threshold):
    col_corr = set()
    corr_mat = data.corr()
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if i != j:
                if abs(corr_mat.iloc[i,j]) >= threshold and corr_mat.columns[i] not in col_corr:
                    if (data.iloc[:,i].corr(df['good_bad'])) <= (data.iloc[:,j].corr(df['good_bad'])):
                        col_name = corr_mat.columns[i]
                    else:
                        col_name = corr_mat.columns[j]
                    col_corr.add(col_name)
                
    return col_corr


# In[77]:


lst_drop = correlation(num[top_num_col], 0.8)


# In[78]:


num = num[top_num_col]


# In[79]:


num.shape


# In[80]:


lst_drop


# In[81]:


num.drop(lst_drop, axis = 1, inplace = True)


# In[82]:


num.head()


# In[83]:


num_col = num.columns.values.tolist()


# ## Outlier Treatment

# In[84]:


#Box Plot Creation 
plt.figure(figsize=(20, 50))
for i in range(len(num_col)):
    plt.subplot(12, 3, i + 1)
    sns.boxplot(data=num, x=num_col[i])
    plt.title(f'Box Plot of {num_col[i]}')

plt.suptitle('Distribution of numerical columns')
plt.tight_layout()


# In[85]:


num['mths_since_last_delinq'].quantile(0.75)


# In[2]:


## Using Inter Quartile range find Upper and Lower limit outside which outlier will need treatment


# In[3]:


# Value below Lower range will be replaced by Lower IQR and above Upper will be replaced by Upper IQR


# In[86]:


for col in num_col:
    q75 = num[col].quantile(0.75)
    q25 = num[col].quantile(0.25)
    upper = q75 + 1.5 * (q75-q25)
    lower = q25 - 1.5 * (q75-q25)
    num[col] = np.where(num[col] > upper, upper, num[col])
    num[col] = np.where(num[col] < lower, lower, num[col])


# In[87]:


#Box plot after IQR treatment
plt.figure(figsize=(20, 50))
for i in range(len(num_col)):
    plt.subplot(12, 3, i + 1)
    sns.boxplot(data=num, x=num_col[i])
    plt.title(f'Box Plot of {num_col[i]}')

plt.suptitle('Distribution of numerical columns')
plt.tight_layout()


# In[88]:


num.shape


# ## Weight Of Evidence & Information value for Categorical variable

# In[89]:


def woe_discrete(df1,variable_name,df_target):
    df1 = pd.concat([df1[variable_name],df_target],axis=1)
    df1 = pd.concat([df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].count()
                     ,df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].mean()],axis=1)
    df1 = df1.iloc[:,[0,1,3]]
    df1.columns = [df1.columns.values[0],'n_obs','prop_good']
    
    df1['prop_n_obs'] = df1['n_obs']/df1['n_obs'].sum()
    df1['n_good']= df1['prop_good'] * df1['n_obs']
    df1['n_bad']= df1['n_obs']-df1['n_good'] #(1-df1['prop_good'])*df1['n_obs']

    df1['prop_n_good'] = df1['n_good']/df1['n_good'].sum()
    df1['prop_n_bad'] = df1['n_bad']/df1['n_bad'].sum()

    df1['WoE'] = np.log(df1['prop_n_good']/df1['prop_n_bad'])

    df1 = df1.sort_values('WoE')
    df1 = df1.reset_index(drop=True)

    df1['diff_prop_good'] = df1['prop_good'].diff().abs()
    df1['diff_WoE'] = df1['WoE'].diff().abs()

    df1['IV'] = ((df1['prop_n_good']-df1['prop_n_bad'])*df1['WoE'])    
    #df1['IV'] = df1['IV'].sum()
    return df1


# In[90]:


def plot_woe(df,rotation_x_axis=0):
    x = np.array(df.iloc[:,0].apply(str))
    y = df['WoE']
    
    plt.figure(figsize=(18,6)) # Sets the graph size to width 18 x height 6.
    
    # Plots the datapoints with coordiantes variable x on the x-axis and variable y on the y-axis.
    plt.plot(x,y,marker='o',linestyle='--',color='g')
    plt.title("Weight of evidence by "+df.columns[0])
    plt.xlabel(df.columns[0]+" categories")  # Names the x-axis with the name of the column with index 0.
    plt.ylabel("WoE") # Names the y-axis 'Weight of Evidence'.
    
    # Rotates the labels of the x-axis a predefined number of degrees.
    plt.xticks(rotation = rotation_x_axis)    
    
    plt.show()


# In[91]:


def woe_categorical(df1, col, y):
    df1 = pd.concat([df1[col], y], axis = 1)
    df1 = pd.concat([df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].count()
                     ,df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].mean()],axis=1)
    df1 = df1.iloc[:,[0,1,3]]
    df1.columns = [df1.columns.values[0],'n_obs','prop_good']
    
    df1['prop_n_obs'] = df1['n_obs']/df1['n_obs'].sum()
    df1['n_good']= df1['prop_good'] * df1['n_obs']
    df1['n_bad']= df1['n_obs']-df1['n_good']
    
    df1['prop_n_good'] = df1['n_good']/df1['n_good'].sum()
    df1['prop_n_bad'] = df1['n_bad']/df1['n_bad'].sum()
    
    df1['WoE'] = np.log(df1['prop_n_good']/df1['prop_n_bad'])
    
    df1 = df1.sort_values('WoE')
    df1 = df1.reset_index(drop=True)
    
    df1['diff_prop_good'] = df1['prop_good'].diff().abs()
    df1['diff_WoE'] = df1['WoE'].diff().abs()

    df1['IV'] = ((df1['prop_n_good']-df1['prop_n_bad'])*df1['WoE'])    
    #df1['IV'] = df1['IV'].sum()
    return df1


# In[92]:


sns.set()


# In[93]:


df_temp = woe_categorical(cat, 'grade', y)
df_temp


# In[94]:


plot_woe(df_temp)


# In[95]:


grade_dict = {}
for i in df_temp.index:
    grade_dict[df_temp.iloc[i, 0]] = df_temp.loc[i, 'WoE']
    print(df_temp.iloc[i, 0],df_temp.loc[i, 'WoE'])


# In[ ]:





# In[96]:


cat.columns


# In[97]:


for col in cat.columns:
    df_temp = woe_categorical(cat, col, y)
    print(df_temp)
    plot_woe(df_temp)
    dict = {}
    for i in df_temp.index:
        dict[df_temp.iloc[i, 0]] = df_temp.loc[i, 'WoE']
    cat[col+'_'] = cat[col].map(dict)
    cat[col+'_'] = np.where(cat[col+'_'].isin([np.inf, -np.inf]), cat[col+'_'].min(), cat[col+'_'])
    cat.drop(col, axis = 1, inplace = True)


# In[98]:


cat


# In[99]:


num.columns


# ## WOE for COntinuous variable

# In[100]:


def woe_ordered_continuous(df, discrete_variabe_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


# In[101]:


num_cat_col = []
del_num_col = []
for col in num.columns:
    #print(col, num[col].nunique())
    #print(col)
    if num[col].nunique() == 1:
        num.drop(col, axis = 1, inplace = True)
        del_num_col.append(col)
    elif num[col].nunique() <= 5:
        num_cat_col.append(col)


# In[102]:


num_cat_col, del_num_col


# In[103]:


num1 = num.copy()

for col in num_cat_col:
    df_temp = woe_categorical(num, col, y)
    print(df_temp)
    plot_woe(df_temp)
    dict = {}
    for i in df_temp.index:
        dict[df_temp.iloc[i, 0]] = df_temp.loc[i, 'WoE']
    
    num1[col+'_'] = num1[col].map(dict)
    num1[col+'_'] = np.where(num1[col+'_'].isin([np.inf, -np.inf]), num1[col+'_'].min(), num1[col+'_'])
    num1.drop(col, axis = 1, inplace = True)
    num.drop(col, axis = 1, inplace = True)


# In[104]:


num.head()


# In[105]:


num1.head()


# In[106]:


num.shape, num1.shape


# ## Replacing all variable with WOE based on WOE, calculated as above

# In[107]:


for col in num.columns:
    num[col+'_factor'] = pd.cut(num[col], 15)
    df_temp = woe_ordered_continuous(num, col+'_factor', y)
    print(df_temp)
    plot_woe(df_temp)
    dict = {}
    for i in df_temp.index:
        dict[df_temp.iloc[i, 0]] = df_temp.loc[i, 'WoE']
    num1[col+'_'] = num1[col].map(dict)
    num1[col+'_'] = np.where(num1[col+'_'].isin([np.inf, -np.inf]), num1[col+'_'].min(), num1[col+'_'])
    num1.drop(col, axis = 1, inplace = True)
    num.drop(col, axis = 1, inplace = True)


# In[108]:


num.head()


# In[109]:


num1.head()


# In[110]:


num.shape, num1.shape


# In[111]:


num1.isnull().mean()


# In[112]:


#num1.dropna(inplace = True)


# In[113]:


num1.shape


# In[114]:


cat.head()


# In[115]:


num1.shape, cat.shape


# In[116]:


df_final = pd.concat([num1, cat, df['good_bad']], axis = 1)


# In[117]:


df_final.shape


# In[118]:


df_final.dropna(inplace = True)


# In[4]:


## Replacing and Removing all unneccesary values and data preparation


# In[119]:


df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna(axis=0)


# In[120]:


df_final.head()


# In[121]:


x = df_final.drop('good_bad', axis = 1)
y = df_final['good_bad']


# In[122]:


df_final.head()


# ## Train Test Split for Model building

# In[123]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, stratify = y)


# In[124]:


xtrain.shape, xtest.shape, ytrain.shape, ytest.shape


# ## Model Building using Logistic regression

# In[125]:


reg = LogisticRegression(max_iter = 1000, class_weight = "balanced")


# In[126]:


cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = 0)


# In[127]:


scores = cross_val_score(reg, xtrain, ytrain, scoring = 'roc_auc', cv = cv)


# In[128]:


reg1 = LogisticRegression(max_iter = 1000, class_weight = "balanced")


# In[129]:


AUROC = np.mean(scores)
GINI = AUROC * 2 - 1

# print the mean AUROC score and Gini
print('Mean AUROC: %.4f' % (AUROC))
print('Gini: %.4f' % (GINI))


# In[130]:


reg.fit(xtrain, ytrain)


# In[131]:


np.transpose(reg.coef_)


# In[132]:


summary = pd.DataFrame(columns = ['Feature Name'], data = xtrain.columns.values)


# In[133]:


summary['Coefficient'] = np.transpose(reg.coef_)


# ## Summary of all features

# In[134]:


summary


# In[135]:


summary.index = summary.index + 1


# In[136]:


summary.loc[0] = ['Intercept', reg.intercept_[0]]


# In[137]:


summary.sort_index(inplace = True)


# In[138]:


y_hat = reg.predict(xtest)


# In[139]:


y_hat_prob = reg.predict_proba(xtest)


# In[140]:


y_hat_prob


# In[141]:


y_hat_prob = y_hat_prob[:][:,1]


# In[142]:


y_val = ytest.copy()


# In[143]:


y_val.reset_index(drop = True, inplace = True)


# In[144]:


test_res = pd.concat([pd.DataFrame(y_hat_prob), y_val], axis = 1)
test_res


# In[145]:


test_res.columns = ['Predicted Test Result', 'Actual Test value']


# In[146]:


test_res


# ## Deciding threshold = 0.25. If < 0.25 then 0 else 1

# In[147]:


tr = 0.25

test_res['y_final'] = np.where(test_res['Predicted Test Result'] > tr, 1, 0)


# In[148]:


test_res['y_final'].head()


# ## Confusion Matrix, ROC Curve, FPR, TPR calculation

# In[149]:


confusion_matrix(test_res['y_final'], test_res['Actual Test value'], normalize = 'all')


# In[150]:


roc_curve(test_res['Actual Test value'],test_res['Predicted Test Result'])


# In[151]:


fpr, tpr, thresholds = roc_curve(test_res['Actual Test value'],
                                 test_res['Predicted Test Result'])
fpr, tpr, thresholds


# In[152]:


test_res.shape


# In[153]:


test_res.head()


# In[154]:


# plot the ROC curve
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')


# In[155]:


precision, recall, thresholds = precision_recall_curve(test_res['Actual Test value'],
                                 test_res['Predicted Test Result'])


# In[156]:


precision, recall, thresholds


# In[157]:


no_skill = len(ytest[ytest == 1]) / len(y)
no_skill


# In[158]:


plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

plt.plot(recall, precision, marker='.', label='Logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('PR curve')


# ## ROC and Gini calculation

# In[159]:


AUCROC = roc_auc_score(test_res['Actual Test value'], test_res['y_final'])


# In[160]:


AUCROC


# In[161]:


Gini = AUROC * 2 - 1
Gini


# In[162]:


summary


# # Scorecard creation

# In[163]:


min_score = 300
max_score = 850


# In[164]:


xtrain


# In[165]:


test_res


# ## Score calculation using following 2 conditions

# In[166]:


test_res['odds'] = (1 - test_res['Predicted Test Result']) / test_res['Predicted Test Result']


# In[167]:


test_res['log_odds'] = np.log(test_res['odds'])


# In[168]:


test_res['log_odds'].min(), test_res['log_odds'].max()


# Assuming, Score of 600 corresponds to odds ratio 50:1 & Increment n score by 40 points result in odds ratio 100:1
# 
# As, Score = Offset + Factor * ln(odds)
# 
# On solving using above both conditions we will get,

# In[169]:


# Using these 2, calculate score for all
factor = 57.14
offset = 376.58


# In[170]:


test_res['Score'] = offset + (factor * test_res['log_odds'])


# In[171]:


act_min = test_res['Score'].min()
act_max = test_res['Score'].max()


# In[172]:


max_score = 950
min_score = 250


# ## Logic to change score range

# In[173]:


act_min, act_max


# In[174]:


test_res['Score_actual'] = ((test_res['Score'] - act_min) / (act_max - act_min) * (max_score - min_score) + min_score)


# In[175]:


test_res['Score_actual'].min(), test_res['Score_actual'].max()


# In[176]:


df_final.head()


# In[177]:


summary


# In[178]:


df_score = pd.concat([xtest, ytest], axis = 1)


# In[179]:


df_score.shape


# In[180]:


df_score.reset_index(inplace = True, drop = True)


# In[181]:


df_score = df_score.loc[:50000,:]
df_score.shape


# In[182]:


ytest_new = df_score['good_bad']
df_score_new = df_score.drop('good_bad', axis = 1)
ytest_new.shape, df_score_new.shape


# In[183]:


col_list = df_score_new.columns.tolist()
col_list


# # Method 2
# 
# ## Score calculation based on individual variable, using Offset and Factor calculation

# In[184]:


for i in range(df_score_new.shape[0]):
    for j in range(df_score_new.shape[1]):
        var = col_list[j]
        #idx = summary('Feature Name').loc[lambda x: x == var].index
        #print(var)
        idx = summary[summary['Feature Name'] == var].index[0]
        #print(idx)
        df_score_new.iloc[i,j] = ((offset / 22) + ((df_score.iloc[i,j] * summary.loc[idx, 'Coefficient']) + (summary.loc[0,'Coefficient'] / 22)) * factor)


# In[185]:


df_score_new.head()


# In[186]:


df_score.head()


# In[187]:


#df_score.drop('Score', axis = 1, inplace = True)


# In[188]:


#df_score['Score'] = df_score.sum(axis = 1)


# In[189]:


#df_score['Score'] = round(df_score['Score'],0)


# In[190]:


#df_score['Score'].min(), df_score['Score'].max()


# ## Score calculation based on score individual variable has received for each variable

# In[191]:


df_score_new['Score'] = df_score_new.sum(axis = 1)


# In[192]:


df_score_new['Score'].min(), df_score_new['Score'].max()


# In[193]:


max_score = 950
min_score = 250


# ## Neutral Score
# Neutral score is most important parameter to judge individual performance of each account based on every parameter 
# and help team to identify where a Customer lag and where he can improvise to get effective loan.
# 
# If a customer has majority of variable score below Neutral Score we can identify him as potential High risk customer,
# though this can be subject to further analysis. But customer with more red flag could be customer with low score and high PD

# In[194]:


neutral_score = ((summary.loc[0,'Coefficient'] / 22) * factor) + (offset / 22)


# In[195]:


neutral_score


# ### Red flag creation

# In[196]:


df_score_new['Red_Flag'] = 0


# In[197]:


for i in range(df_score_new.shape[0]):
    count = 0
    for j in range(df_score_new.shape[1]):
        if df_score_new.iloc[i,j] <= neutral_score:
            count += 1
    df_score_new.loc[i, 'Red_Flag'] = count
    #df_score.loc[i, 'Red_Flag'] = df_score[df_score.loc[i,:] > neutral_score].count()


# In[198]:


df_score_new.head()


# #### Red flag distribution

# In[199]:


df_score_new['Red_Flag'].min(), df_score_new['Red_Flag'].max()


# In[200]:


df_score_new.loc[df_score_new['Red_Flag'] > 15].shape


# In[201]:


df_final = pd.concat([df_score_new, ytest_new], axis = 1)
df_final.head()


# In[202]:


df_final['good_bad_rev'] = np.where(df_final['good_bad'] == 1, 0, 1)


# In[203]:


df_final.tail()


# ## Scorecard validation using AUCROC and GINI calculation

# In[204]:


#Sort data in descending order of score or ascending order of Probability of Default
df_final.sort_values(by = 'Score', inplace = True)


# In[205]:


df_final['good_bad_rev'].sum(), df_final['good_bad'].sum()


# In[206]:


df_final['cum_sum'] = df_final['good_bad_rev'].cumsum(axis = 0)


# In[207]:


df['cum_total'] = 0


# In[208]:


df_final.reset_index(inplace = True, drop = True)


# In[209]:


count = 1
for i in range(df_final.shape[0]):
    df_final.loc[i, 'cum_total'] = count + i


# ### Total Bad and Total Good Calculation

# In[210]:


df_final['cum_bad_prop'] = df_final['cum_sum'] / df_final['good_bad_rev'].sum()
df_final['cum_total_prop'] = df_final['cum_total'] / df_final.shape[0]


# In[211]:


df_final['AUC'] = 0


# #### Calculation of AUC and GINI for score allotment

# In[212]:


for i in range(df_final.shape[0]):
    if i == 0:
        df_final.loc[i, 'AUC'] = 0.5 * df_final.loc[i, 'cum_bad_prop'] * df_final.loc[i, 'cum_total_prop']
    else:
        df_final.loc[i, 'AUC'] = 0.5 * (df_final.loc[i, 'cum_bad_prop'] + df_final.loc[i-1, 'cum_bad_prop']) * (df_final.loc[i, 'cum_total_prop'] - df_final.loc[i-1, 'cum_total_prop'])


# In[213]:


df_final['AUC'].sum()


# In[214]:


Gini = (2*df_final['AUC'].sum() - 1) * 100


# In[215]:


Gini


# # Result & Conclusion

#  1. Above result suggest, model able to provide lower score for customer with high PD
#  2. Model able to bifurcate beteween Good and Bad easily based on result
#  3. There can be further varification to be done to evaluate Scorecard
#  4. bank can have more control over each loan as score has been assigned on variable level hence each variable could be controlled individually

# # Future Scope

# 1. More refined approach for Missing value treatment
# 2. Defined statistical approach for variable selection, currently too manual and less staistical
# 3. Outlier treatment could be more refined
# 4. More emphasis could be provided to remove, Multi collinearity, Homoskedasticity, Normality of variables and many others
# 5. More Robust models like Decision Tree, Random Forest or Boosting Algorithms could provide better result
# 6. Variables have been replaced with WOE values, this could be avoided and model could be built with Original values, this method can be tested
# 

# In[ ]:




