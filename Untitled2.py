#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# In[2]:


data=pd.read_csv(r"C:\Users\APEKSHA  PRATIKSHA\OneDrive\Desktop\spam.csv", encoding='latin1')


# In[3]:


data.head()


# In[4]:


data.info()


# In[6]:


data.isna().sum()


# In[12]:


data['v1'].value_counts()


# In[16]:


import nltk
nltk.download('omw-1.4')


# In[18]:


corpus = []
lm = WordNetLemmatizer()
for i in range (len(data)):
    review = re.sub('^a-zA-Z0-9',' ',data['v2'][i])
    review = review.lower()
    review = review.split()
    review = [data for data in review if data not in stopwords.words('english')]
    review = [lm.lemmatize(data) for data in review]
    review = " ".join(review)
    corpus.append(review)    


# In[20]:


data['v2'][0]


# In[21]:


len(data['v2'])


# In[22]:


len(corpus)


# In[23]:


data['v2']=corpus
data.head()


# In[24]:


x = data['v1']
y = data['v2']


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 10)


# In[26]:


len(x_train), len(y_train)


# In[27]:


len(x_test),len(y_test)


# In[28]:


tf_obj = TfidfVectorizer()
x_train_tfidf = tf_obj.fit_transform(x_train).toarray()
x_train_tfidf


# In[29]:


x_train_tfidf.shape


# In[30]:


text_mnb = Pipeline([('tfidf',TfidfVectorizer()),('mnb',MultinomialNB())])


# In[31]:


text_mnb.fit(x_train,y_train)


# In[32]:


y_pred_test = text_mnb.predict(x_test)
print("Accuracy Score:", accuracy_score(y_test,y_pred_test)*100)


# In[33]:


y_pred_train = text_mnb.predict(x_train)
print("Accuracy Score:",accuracy_score(y_train,y_pred_train)*100)


# In[34]:


y_pred_test = text_mnb.predict(x_test)
print("Confusion Matrix on Test Data:\n", confusion_matrix(y_test,y_pred_test))


# In[35]:


y_pred_test = text_mnb.predict(x_test)
print("Classification Reportx on Test Data:\n", classification_report(y_test,y_pred_test))


# In[36]:


def preprocess_data(text):
    review = re.sub('^a-zA-Z0-9',' ',text)
    review = review.lower()
    review = review.split()
    review = [data for data in review if data not in stopwords.words('english')]
    review = [lm.lemmatize(data) for data in review]
    review = " ".join(review)
    return [review]


# In[38]:


user_data = data['v2'][0]
print(user_data)
user_data = preprocess_data(user_data)
user_data


# In[39]:


text_mnb.predict(user_data)[0]


# In[40]:


class prediction:
    
    def __init__(self,data):
        self.data = data
        
    def user_data_preprocessing(self):
        lm = WordNetLemmatizer()
        review = re.sub('^a-zA-Z0-9',' ',self.data)
        review = review.lower()
        review = review.split()
        review = [data for data in review if data not in stopwords.words('english')]
        review = [lm.lemmatize(data) for data in review]
        review = " ".join(review)
        return [review]
    
    def user_data_prediction(self):
        preprocess_data = self.user_data_preprocessing()
        
        if text_mnb.predict(preprocess_data)[0] == 'spam':
            return 'This Message is Spam'
            
        else:
            return 'This Message is Ham'  


# In[41]:


data.head()


# In[42]:


user_data = data['v2'][2]
print(user_data)
prediction(user_data).user_data_prediction()


# In[43]:


user_data = data['v2'][3]
print(user_data)
prediction(user_data).user_data_prediction()


# In[ ]:




