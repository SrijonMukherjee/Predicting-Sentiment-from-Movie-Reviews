#!/usr/bin/env python
# coding: utf-8

# # Importing Required Libraries

# In[ ]:


# Importing Required Libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt


# # Importing and Analyzing the Dataset

# In[ ]:


# Import files from your computer
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))


# In[ ]:


# Import the dataset
movie_reviews_labeled = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

movie_reviews_unlabeled = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

test_data = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)


# In[ ]:


# Convert .tsv file to .csv file and read the csv file 
movie_reviews_labeled.to_csv('movie_reviews_labeled.csv')
movie_reviews = pd.read_csv('movie_reviews_labeled.csv')


# In[ ]:


# Remove quotations from string and print first 5 rows of the dataset
movie_reviews['review'] = movie_reviews['review'].str.strip('" "')
movie_reviews.head()


# In[ ]:


# Take a look at any one of the reviews
movie_reviews['review'][0]


# In[ ]:


# See the size of positive and negative sentiments in this dataset
movie_reviews.shape


# From the output, it is clear that the dataset contains equal number of positive and negative reviews.

# # Data preprocessing

# In[ ]:


# Take a text string as a parameter
# Performs preprocessing on the string to remove special chracters from the string

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)


# In[ ]:


# Preprocess our reviews and will store them in a new list as shown below
X = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(preprocess_text(sen))

X[0]


# In[ ]:


y = movie_reviews['sentiment']


# In[ ]:


# Divide the dataset into 80% for training set and 20% for testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# # Preparing the Embedding Layer

# In[ ]:


# Prepare the embedding layer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# In[ ]:


# Find the vocabulary size and then perform padding on both train and test set
# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# In[ ]:


# Load the GloVe word embeddings
# Create a dictionary that will contain words as keys and their corresponding embedding list as values.
from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()


# In[ ]:


# Create an embedding matrix where each row number will correspond to the index of the word in the corpus
embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# ### Text Classification with a Convolutional Neural Network

# In[ ]:


# Create a simple convolutional neural network with 1 convolutional layer and 1 pooling layer
from keras.layers.convolutional import Conv1D
model = Sequential()

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
model.add(embedding_layer)

model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())


# In[ ]:


# Train our model and evaluate it on the training set
history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.33)

score = model.evaluate(X_test, y_test, verbose=1)


# In[ ]:


print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()


# ### Configuring Learning Rate and Batch Size

# #### Low Batch Size

# In[ ]:


# Create a simple convolutional neural network with 1 convolutional layer and 1 pooling layer
from keras.layers.convolutional import Conv1D
model = Sequential()

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
opt = tensorflow.keras.optimizers.Adam(learning_rate = 0.01)
model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=opt)

print(model.summary())


# In[ ]:


from tensorflow.keras.callbacks import LearningRateScheduler
# Train our model and evaluate it on the training set
history = model.fit(X_train, y_train, batch_size=32, epochs=15, verbose=1, validation_split=0.33)

score = model.evaluate(X_test, y_test, verbose=1)


# In[ ]:


print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()


# #### High Batch Size

# In[ ]:


# Create a simple convolutional neural network with 1 convolutional layer and 1 pooling layer
from keras.layers.convolutional import Conv1D
model = Sequential()

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
opt = tensorflow.keras.optimizers.Adam(learning_rate = 0.1)
model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=opt)

print(model.summary())


# In[ ]:


from tensorflow.keras.callbacks import LearningRateScheduler
# Train our model and evaluate it on the training set
history = model.fit(X_train, y_train, batch_size=256, epochs=15, verbose=1, validation_split=0.33)

score = model.evaluate(X_test, y_test, verbose=1)


# In[ ]:


print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()


# ### Text Classification with Recurrent Neural Network (LSTM)

# In[ ]:


# Use an LSTM (Long Short Term Memory network)
from keras.layers import LSTM

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(LSTM(128))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# In[ ]:


print(model.summary())


# In[ ]:


history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)


# In[ ]:


print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# # Making Predictions on Single Instance

# In[ ]:


# Select a review from our corpus
instance = X[57]
print(instance)


# In[ ]:


# Convert this review into numeric form to predict the sentiment
instance = tokenizer.texts_to_sequences(instance)

flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)

flat_list = [flat_list]

instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)

model.predict(instance)


# The sentiment is predicted as positive.

# In[ ]:




