#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#for the visualization
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)


# In[2]:


#reading the dataset file through pandas library
data = pd.read_csv('\Seed_Data.csv')


# In[3]:


#random 5 samples out of the dataset
data.sample(5)


# In[4]:


#information about the data
data.info()


# In[5]:


#describing the data as COUNT-MEAN
print(data.describe())


# In[6]:


#visualization
#heatmap for the correlation explains the relationship for each with the features

plt.figure(figsize=[8,8])
sns.heatmap(data.corr(), annot=True, cmap="YlGnBu")
plt.title('Correlations of the Features')
plt.show()


# In[7]:


#the dataset has 3 values in 'target': rosa, kama, canadian (type of seed)
sns.countplot(data['target'], palette='husl')
plt.show()


# In[8]:


#visualizing the relationship between target and AREA
#when the target = 0 when the AREA is from 12.5 to 16.5...etc
a = sns.FacetGrid(data, col='target')
a.map(sns.boxplot, 'A', color='yellow', order=['0', '1', '2'])


# In[9]:


#visualizing the relationship between target and PARAMETER
p = sns.FacetGrid(data, col='target')
p.map(sns.boxplot, 'P', color='orange', order=['0', '1', '2'])


# In[10]:


c = sns.FacetGrid(data, col='target')
c.map(sns.boxplot, 'C', color='red', order=['0', '1', '2'])


# In[11]:


lk = sns.FacetGrid(data, col='target')
lk.map(sns.boxplot, 'LK', color='purple', order=['0', '1', '2'])


# In[12]:


wk = sns.FacetGrid(data, col='target')
wk.map(sns.boxplot, 'WK', color='blue', order=['0', '1', '2'])


# In[13]:


acoef = sns.FacetGrid(data, col='target')
acoef.map(sns.boxplot, 'A_Coef', color='cyan', order=['0', '1', '2'])


# In[14]:


lkg = sns.FacetGrid(data, col='target')
lkg.map(sns.boxplot, 'LKG', color='green', order=['0', '1', '2'])


# In[15]:


wk = sns.FacetGrid(data, col='target')
wk.map(sns.boxplot, 'WK', color='blue', order=['0', '1', '2'])


# In[16]:


acoef = sns.FacetGrid(data, col='target')
acoef.map(sns.boxplot, 'A_Coef', color='cyan', order=['0', '1', '2'])


# In[17]:


lkg = sns.FacetGrid(data, col='target')
lkg.map(sns.boxplot, 'LKG', color='green', order=['0', '1', '2'])


# In[18]:


#Split-out validation dataset 80 train , 20 test
array = data.values
x = array[:,0:7]


# In[19]:


#target is Y
y = array[:,7]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[23]:


#K-NN Algorithm
from sklearn.neighbors import KNeighborsClassifier
import math
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print(round(float(sum(pred==y_test)/len(y_test)),2))


# In[21]:


#Testing accuracy
from sklearn.metrics import accuracy_score
print('Test Accuracy Score:', accuracy_score(y_test,pred))


# In[22]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[ ]:





# In[ ]:




