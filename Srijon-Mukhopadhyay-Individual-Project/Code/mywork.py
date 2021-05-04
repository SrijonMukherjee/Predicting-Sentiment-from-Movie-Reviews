pred_cnn = model.predict(X_test)
pred_cnn = (pred_cnn>0.5).astype(int)

import seaborn as sns
conf_mat = confusion_matrix(y_test, pred_cnn)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=['Negative','Positive'], yticklabels = ['Negative','Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('cnn_cm', dpi = 720)
plt.show()

report = metrics.classification_report(y_test, pred_cnn)

from sklearn.metrics import roc_curve, roc_auc_score
pred_cnn = model.predict(X_test)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, pred_cnn)


from sklearn.metrics import auc

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Area Under Curve (AUC) =  {:.3f}'.format(auc(fpr,tpr)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve for Predicting Sentiments')
plt.legend(loc='best')
plt.savefig('auc_cnn.jpg', dpi = 720)
plt.show()


from tensorflow.keras.callbacks import LearningRateScheduler
initial_learning_rate = 0.01
epochs = 5
decay = initial_learning_rate / epochs
def lr_time_based_decay(epoch, lr):
  lrate = lr * 1 / (1 + decay * epoch)
  return lrate

lrate = LearningRateScheduler(lr_time_based_decay, verbose = 1)

# Create a Bidirectional LSTM with low learning rate

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, SpatialDropout1D
from tensorflow.keras.layers import Embedding, Flatten
from tensorflow.keras.layers import MaxPooling1D, Dropout, Activation, Conv1D

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
opt = tensorflow.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

print(model.summary())


history = model.fit(X_train, y_train, batch_size=32, epochs=15, verbose=1, validation_split=0.33)


print("Test Score:", score[0])
print("Test Accuracy:", score[1])


pred_lstm = model.predict(X_test)

pred_label_lstm = (pred_lstm>0.5).astype(int)

report = metrics.classification_report(y_test, pred_label_lstm)
print(report)



import seaborn as sns
conf_mat= confusion_matrix(y_test, pred_label_lstm)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=['Negative','Positive'], yticklabels = ['Negative','Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('lstm_cm.jpg', dpi = 720)
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score
pred_lstm = model.predict(X_test)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, pred_lstm)



from sklearn.metrics import auc

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Area Under Curve (AUC) =  {:.3f}'.format(auc(fpr,tpr)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve for Predicting Positive Sentiments')
plt.legend(loc='best')
plt.savefig('auc_lstm.jpg', dpi = 720)
plt.show()



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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, SpatialDropout1D
from tensorflow.keras.layers import Embedding, Flatten
from tensorflow.keras.layers import MaxPooling1D, Dropout, Activation, Conv1D

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(SpatialDropout1D(0.3))
model.add(LSTM(128,recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
opt = tensorflow.keras.optimizers.Adam(learning_rate = 0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

print(model.summary())


from tensorflow.keras.callbacks import LearningRateScheduler
# Train our model and evaluate it on the training set
history = model.fit(X_train, y_train, batch_size=128, epochs=15, verbose=1, validation_split=0.33)

score = model.evaluate(X_test, y_test, verbose=1)


print("Test Score:", score[0])
print("Test Accuracy:", score[1])


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

