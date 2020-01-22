#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import norm
from matplotlib import pyplot as plt
#import tensorflow as tf
import seaborn as sns
sns.set()


# In[2]:


def kl_divergence(p, q):
    return np.sum(np.where(np.logical_and(p != 0, q != 0), p * np.log(p / q), 0))
dataset = pd.read_csv('titanic.csv')
dataset.head()


# In[3]:


dataset.describe()


# In[21]:





# In[6]:


dataset['Age'].fillna(dataset['Age'].mode()[0], inplace=True)

female_age = dataset[dataset['Sex'] == 'female']['Age']
total_counts = len(female_age)
data_norm = norm.rvs(size=total_counts, loc=0, scale=50)
count, devision = np.histogram(female_age, bins=5)
count2, devision2 = np.histogram(data_norm, bins=5)

print(kl_divergence(count, count2))

plt.plot(count, c='blue')
plt.plot(count2, c='red')

