#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#importing dataset
spam_df=pd.read_csv(r'C:\Users\MANOJ KUMAR SHRIVAST\Desktop\ml projects\emails.csv')


# In[3]:


spam_df


# In[4]:


spam_df.head()


# In[5]:


spam_df.tail()


# In[7]:


spam_df.describe()


# In[8]:


ham=spam_df[spam_df['spam']==0]


# In[9]:


ham


# In[10]:


spam=spam_df[spam_df['spam']==1]


# In[11]:


spam


# In[12]:


print('Spam Percentage=',(len(spam)/len(spam_df)*100),'%')


# In[13]:


sns.countplot(spam_df['spam'])


# In[14]:


#We will use Count Vectorizer Method to convert the texts into numbers,basically they represent the text into array of numbers
#FIRST FOR SAMPLE DATA
from sklearn.feature_extraction.text import CountVectorizer
sample_data=['this is the first document','this is number second','And this is the third one']
sample_vectorizer=CountVectorizer()
X=sample_vectorizer.fit_transform(sample_data)


# In[15]:


print(X.toarray())


# In[16]:


print(sample_vectorizer.get_feature_names())
#it simply encoded the text into number in corresponding position by their count


# In[17]:


#FOR SPAM/HAM EXAMPLE
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
spamham_countvect=vectorizer.fit_transform(spam_df['text'])


# In[18]:


print(spamham_countvect.toarray())


# In[19]:


print(vectorizer.get_feature_names())


# In[20]:


label=spam_df['spam'].values


# In[21]:


label


# In[22]:


#first train model
from sklearn.naive_bayes import MultinomialNB
Nb_classifier=MultinomialNB()
Nb_classifier.fit(spamham_countvect,label)


# In[23]:


testing_sample=['Free Money!!!','Hi Rishab,please let me know if you need any further info']
testing_sample_countvect=vectorizer.transform(testing_sample)


# In[24]:


test_sample_predict=Nb_classifier.predict(testing_sample_countvect)


# In[26]:


test_sample_predict


# In[27]:


X=spamham_countvect
y=label


# In[28]:


#SPLITTING THE DATA
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[29]:


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train,y_train)


# In[30]:


#Evaluating the Model
from sklearn.metrics import classification_report,confusion_matrix
y_predict_train=classifier.predict(X_train)


# In[31]:


y_predict_train


# In[32]:


cm=confusion_matrix(y_train,y_predict_train)
sns.heatmap(cm,annot=True)


# In[33]:


y_predict_test=classifier.predict(X_test)


# In[34]:


cm=confusion_matrix(y_train,y_predict_train)
sns.heatmap(cm,annot=True)


# In[35]:


cm=confusion_matrix(y_test,y_predict_test)
sns.heatmap(cm,annot=True)


# In[36]:


print(classification_report(y_test,y_predict_test))


# In[ ]:




