#!/usr/bin/env python
# coding: utf-8

# # Fetching Dataset

# In[162]:


from sklearn.datasets import fetch_openml
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[20]:


mnist = fetch_openml('mnist_784')


# In[51]:


x, y= mnist['data'], mnist['target']


# In[62]:


x.shape


# In[ ]:


some_digit = np.array(x.iloc[3600]) #Convert data into Array iloc-> to get specific row&coloumn
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap= matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off") # For removing scale


# In[168]:


y.iloc[3600]


# In[172]:


x_train= np.array(x.iloc[:6000])
x_test = np.array(x.iloc[6000:7000])


# In[173]:


y_train= np.array(y.iloc[:6000])
y_test= np.array(y.iloc[6000:7000])


# In[174]:


shuffle_index = np.random.permutation(6000)
x_train = x_train[shuffle_index]
y_train = y_train[shuffle_index]


# # Creating a 2 Detector

# In[175]:


y_train = y_train.astype(np.int8) #Output is in string '1','2' 
y_test= y_test.astype(np.int8) #It converts it into numbers. String->No.
y_train_2 = (y_train==2)
y_test_2 = (y_test==2)
y_train_2


# In[176]:


y_train


# In[182]:


clf = LogisticRegression(tol=0.1,max_iter=1000,solver='lbfgs')
clf.fit(x_train , y_train_2)


# In[183]:


clf.predict([some_digit])


# # cross validation 

# In[184]:


a= cross_val_score(clf,x_train,y_train_2,cv=3,scoring="accuracy")


# In[185]:


a.mean()


# In[ ]:




