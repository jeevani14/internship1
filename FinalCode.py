#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Importing the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#Reading the csv file using pandas
df = pd.read_csv("complaints_processed.csv")


# In[6]:


#displaying the first five rows
df.head()


# In[7]:


#shape/size of the dataframe
df.shape


# In[8]:


#Finding the null values
df.isna().sum()


# In[9]:


#Dropping the unnamed column
df.drop(["Unnamed: 0"], axis = 1, inplace =True)


# In[10]:


#Dropping NA values
df.dropna(axis=0, inplace=True)


# In[11]:


#Printing the count of each class
df.groupby(["product"]).count()


# In[12]:


#Bar graph for visualization
plt.figure(figsize = (10,5))
sns.countplot(x=df['product'],data=df)
plt.show()


# In[13]:


#Using regular expression for punctuation removal
import re
replace_symbols = re.compile('[/(){}\[\]\|@,;]')
Replace_signs = re.compile('[^0-9a-z #+_]')


# In[14]:


#Pre processing by defining a function and converting to lowercase, removing stopwords and punctuations
def cleaning(text):
  text = text.lower()
  text = replace_symbols.sub(' ', text)
  text = Replace_signs.sub(' ', text)
  text = text.replace('x','')
  #text = ' '.join(word for word in text.split() if word not in STOPWORDS)

  return text
df['narrative'] = df['narrative'].apply(cleaning)


# In[15]:


df['narrative']


# In[16]:


df.tail(10)


# In[17]:


#Remving the last 6 rows
df.drop(df.tail(6).index, inplace=True)


# In[18]:


#Dummification 
df = df.join(pd.get_dummies(df['product']))


# In[19]:


#Droping product column
df.drop(['product'], axis=1, inplace=True)


# In[20]:


df.head()


# In[21]:


#Stopword removal
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[22]:


stop_words = stopwords.words('english')


# In[23]:


df['narrative'] = df['narrative'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))


# In[24]:


df['narrative']


# In[25]:


#Importing necssary libraries for lemmatization
import nltk
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

# Initializing the lemmatizer
lemmatizer = WordNetLemmatizer()


# In[26]:


#Defining a function for lemmatization
def lemmatize_words(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word,pos='v') for word in words]
    return ' '.join(words)
df['narrative'] = df['narrative'].apply(lemmatize_words)


# In[29]:


#Tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
t = Tokenizer(num_words = 500)
t.fit_on_texts(df['narrative'])


# In[30]:


df['narrative']


# In[31]:


df['word count'] = [len(i.split(' ')) for i in df['narrative']]


# In[32]:


sent_length = df['word count'].max()


# In[33]:


sent_length


# In[61]:


# saving the tokenizer for predict function
pickle.dump(t, open('token.pkl', 'wb'))


# In[35]:


#Coverting to sequences 
text = t.texts_to_sequences(df['narrative'])


# In[36]:


#Padding the sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
x=pad_sequences(text,padding='post',maxlen=250)
print(x)


# In[37]:


x = np.array(x)


# In[38]:


df.head(1)


# In[39]:


df.drop(['word count'], axis=1, inplace=True)


# In[40]:


#Target variable
y=df.iloc[:,1:].to_numpy()
y


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


#Splitting the data into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[44]:


X_train


# In[45]:


#Randomforest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


rfc = RandomForestClassifier(n_estimators=150, criterion='entropy', class_weight = 'balanced', random_state=42)


# In[ ]:


#Fitting the data
rfc.fit(X_train, Y_train)


# In[ ]:


#Prediction
y_pred = rfc.predict(X_test)


# In[ ]:


#Finding the accuracy
accuracy = accuracy_score(Y_test, y_pred)
print('Accuracy:', accuracy)


# In[ ]:


#Classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))


# In[66]:


from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 500
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

from keras.regularizers import l1_l2

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1], 
                    embeddings_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
#model.add(SpatialDropout1D(0.2))
# model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.3, return_sequences=True))
# model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.3, return_sequences=True))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.3))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[49]:


from tensorflow.keras.callbacks import ModelCheckpoint
# creating check point and saving in file named nextwords.h5
checkpoint = ModelCheckpoint("classification.h5", monitor='loss', verbose=1, save_best_only=True)


# In[50]:


result = model.fit(X_train, Y_train, batch_size=64, epochs=5, callbacks=[checkpoint])


# In[63]:


# Load the saved tokenizer and model
tokenizer = Tokenizer()
loaded_tokenizer = pickle.load(open('token.pkl', 'rb'))
tokenizer.word_index = loaded_tokenizer.word_index


# In[64]:


model = load_model('classification.h5')

# Define the target labels
labels = ['credit_card', 'credit_reporting', 'debt_collection', 'mortgages_and_loans', 'retail_banking']

# Define a function for text classification
def predict_class(text):
    # Preprocess the text
    text = text.lower()
    text = loaded_tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, padding='post', maxlen=250)
    
    # Predict the class
    prediction = model.predict(text)[0]
    class_index = np.argmax(prediction)
    predicted_class = labels[class_index]
    confidence = prediction[class_index] * 100
    
    return predicted_class, confidence


text = input("Enter the complaint: ")
predicted_class, confidence = predict_class(text)
print('Predicted class:', predicted_class)
print('Confidence:', confidence)


# In[ ]:




