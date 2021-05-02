import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Input, GlobalMaxPooling1D, Dropout, Merge, concatenate
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras import optimizers
import pickle


def plotting(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

do_early_stopping = True
# top words to be considered in Tokenizer
NUM_WORDS = 20000

# Length of phrases for padding if shorter or cropping if longer
MAX_SEQUENCE_LENGTH = 500

EMBEDDING_DIM = 300

# preparing train-set from text data
train_text = np.load(dataset_dir + 'Res/train_text.npy')
train_label = np.load(dataset_dir + 'Res/train_label.npy')

print('TrainSet is composed of %s texts.' % len(train_text))

# preparing test-set from text data
test_text = np.load(dataset_dir + 'Res/test_text.npy')
test_label = np.load(dataset_dir + 'Res/test_label.npy')

print('TestSet is composed of %s texts.' % len(test_text))

# Formatting text samples and labels in tensors.
with open(dataset_dir + 'Res/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

train_sequences = tokenizer.texts_to_sequences(train_text) # Splits words by space (split=” “), Filters out punctuation, Converts text to lowercase. For each text returns a list of integers (same words a codified by same integer)

test_sequences = tokenizer.texts_to_sequences(test_text)
word_index = tokenizer.word_index # dictionary mapping words (str) to their index starting from 0 (int)
print('Found %s unique tokens.' % len(word_index))

train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH) # each element of sequences is cropped or padded to reach maxlen 
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_label = np.asarray(train_label)
test_label = np.asarray(test_label)
print('Shape of data tensor:', train_data.shape)

#shuffle dataset
indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)
train_data = train_data[indices]
train_label = train_label[indices]

# split the data into a training set and a validation set

num_validation_samples = int(0.1 * train_data.shape[0])

x_train = train_data[:-num_validation_samples]
y_train = train_label[:-num_validation_samples]

x_val = train_data[-num_validation_samples:]
y_val = train_label[-num_validation_samples:]

x_test = test_data
y_test = test_label


embedding_matrix = np.load(dataset_dir + 'Res/embedding_matrix.npy')

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_index)+1, EMBEDDING_DIM, weights=[embedding_matrix],
                                               input_length=MAX_SEQUENCE_LENGTH, trainable=False)

x = embedding_layer(sequence_input)
x = Dropout(0.3)(x)
x = Conv1D(200, 5, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = LSTM(100)(x)
x = Dropout(0.3)(x)
prob = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, prob)
optimizer = optimizers.Adam(lr=0.0004)
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])