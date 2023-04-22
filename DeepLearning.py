#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow as tf; print(tf.__version__)
from spacy.lang.en import English

import tensorflow as tf
import datetime

import pandas as pd
from spacy.lang.en import English
import spacy
import re


# In[2]:


df = pd.read_csv("FakeJob.csv")
df.head()


# In[3]:


df['fraudulent'].value_counts()
#Clearly the data is imbalanced


# In[4]:


df.pivot_table(index=['fraudulent'], columns='employment_type', aggfunc='size', fill_value=0)


# In[5]:


df.pivot_table(index=['fraudulent'], columns=['required_experience'], aggfunc='size', fill_value=0)


# In[6]:


df.pivot_table(index=['fraudulent'], columns=['required_education'], aggfunc='size', fill_value=0)


# In[7]:


df.isna().sum()


# In[8]:


df.fillna(' ',inplace=True)


# In[9]:


df['text']=df['title']+" " + df['department'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits'] + " " 


# In[10]:


df.head()


# In[11]:


delete_list=['job_id','title','location','telecommuting','has_company_logo','has_questions','department','salary_range','company_profile','description','requirements','benefits','employment_type','required_experience','required_education','industry','function']

for val in delete_list:
    del df[val]
df.head()


# In[14]:


df['text']=df['text'].str.replace('\n','')
df['text']=df['text'].str.replace('\r','')
df['text']=df['text'].str.replace('\t','')
  
  #This removes unwanted texts
df['text'] = df['text'].apply(lambda x: re.sub(r'[0-9]','',x))
df['text'] = df['text'].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]',' ',x))
  
  #Converting all upper case to lower case
df['text']= df['text'].apply(lambda s:s.lower() if type(s) == str else s)
  

  #Remove un necessary white space
df['text']=df['text'].str.replace('  ',' ')

  #Remove Stop words
nlp=spacy.load("en_core_web_sm")
df['text'] =df['text'].apply(lambda x: ' '.join([word for word in x.split() if nlp.vocab[word].is_stop==False ]))


# In[15]:


sp = spacy.load('en_core_web_sm')
import time
t1=time.time()
output=[]

for sentence in df['text']:
    sentence=sp(str(sentence))
    s=[token.lemma_ for token in sentence]
    output.append(' '.join(s))
df['processed']=pd.Series(output)
t=time.time()-t1
print("Time" + str(t))


# In[16]:


import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 100000
embedding_dim = 64
max_length = 250
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
#Tokenization

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(df['processed'].values)
word_index = tokenizer.word_index
print(len(word_index))


# In[17]:


X = tokenizer.texts_to_sequences(df['processed'].values)                         #Tokenize the dataset
X = pad_sequences(X, maxlen=max_length)     #Padding the dataset
Y=df['fraudulent']                                                                   #Assign the value of y  
print(Y.shape)


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20,random_state=41)


# In[19]:


model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.SpatialDropout1D(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[20]:


import numpy as np
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_train.shape


# In[21]:


pip install keras


# In[22]:


from keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
history = model.fit(X_train, y_train, epochs=10,batch_size=64, validation_split=0.1,callbacks=callbacks, verbose=1)


# In[23]:


import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


# In[24]:


y_predict=model.predict(X_test)
y_predict = np.around(y_predict, decimals = 0)
y_predict


# In[25]:


from sklearn.metrics import confusion_matrix
cf=confusion_matrix(y_test,y_predict)
cf


# In[26]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("classification report of bi-LSTM neural network.")
print("=====================================================")
c_report=classification_report(y_test,y_predict,target_names = ['0','1'])
print(c_report)


# In[27]:


from sklearn.metrics import accuracy_score
sc = accuracy_score(y_test,y_predict)


# In[28]:


sc


# In[29]:


get_ipython().run_line_magic('store', 'sc')


# In[30]:


scores = {''}


# In[ ]:




