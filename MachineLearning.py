#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd

df = pd.read_csv("FakeJob.csv")
df.head()


# In[30]:


df.columns


# In[31]:


del_cols=['job_id','location','department','salary_range','description','title','company_profile','benefits','requirements']


# In[32]:


df=df.drop(del_cols,axis=1)


# In[33]:


col_list=list(df.columns)


# In[34]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[35]:


import numpy as np
for i in col_list:
    if df[i].dtype=='object':
        df[i]=df[i].replace(np.nan,df[i].mode()[0],regex=True)


# In[36]:


df.isnull().sum()


# In[37]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[38]:


for i in col_list:
    if df[i].dtype=='object':
        df[i]=le.fit_transform(df[i])


# In[39]:


y=df['fraudulent']


# In[40]:


x=df.drop(['fraudulent'],axis=1)


# In[41]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr=LogisticRegression()


# In[43]:


lr.fit(x_train,y_train)
pred_1=lr.predict(x_test)


# In[44]:


score_1=accuracy_score(y_test,pred_1)


# In[45]:


score_1


# In[46]:


from sklearn.metrics import classification_report
print("classification report of logistic regression.")
print("=============================================")
print(classification_report(y_test,pred_1))


# In[47]:


from sklearn.neighbors import KNeighborsClassifier
list_1=[]
for i in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_y=knn.predict(x_test)
    scores=accuracy_score(y_test,pred_y)
    list_1.append(scores)


# In[48]:


import matplotlib.pyplot as plt
plt.plot(range(1,11),list_1)
plt.xlabel('k values')
plt.ylabel('accuracy scores')
plt.show()


# In[49]:


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
pred_2=knn.predict(x_test)
score_2=accuracy_score(y_test,pred_2)


# In[50]:


print("classification report of KNN algorithm.")
print("=============================================")
print(classification_report(y_test,pred_2))


# In[51]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)


# In[52]:


pred_4=rfc.predict(x_test)
score_4=accuracy_score(y_test,pred_4)
score_4


# In[53]:


print("classification report of Random Forest algorithm.")
print("=============================================")
print(classification_report(y_test,pred_4))


# In[54]:


from sklearn.metrics import f1_score


# In[55]:


get_ipython().run_line_magic('store', '-r sc')
sc


# In[56]:


scores = {'Logistic_Regression':score_1,'Knn':score_2,'Random_Forest':score_4,'Bi-LSTM':sc}

Algorithms = list(scores.keys())
Accuracy_scores = list(scores.values())

fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(Algorithms, Accuracy_scores, color ='blue',
        width = 0.4)
 
plt.xlabel("Algorithms")
plt.ylabel("Accuracy scores")
plt.title("Accuacr scores of various classification algorithms")
plt.ylim(0.9, 1.0)
plt.show()


# In[ ]:




