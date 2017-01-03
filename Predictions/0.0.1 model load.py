# from __future__ import print_function
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys
import pandas as pd
from keras.models import load_model

# np.random.seed(1337)
#
BASE_DIR = '/Users/kimardenmiller/Dropbox/PyCharm/NLP_CNN/Prediction_New_Inputs'
# GLOVE_DIR = BASE_DIR + 'glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
# EMBEDDING_DIM = 100
# VALIDATION_SPLIT = 0.2

# # first, build index mapping words in the embeddings set to their embedding vector

# print('Indexing word vectors.')
#
# embeddings_index = {}
#
# f = pd.read_csv(os.path.join(GLOVE_DIR, 'glove.6B.100d_2.txt'), sep=" ", header=None).values
# for line in f:
#     word = line[0]
#     coefficients = np.asarray(line[1:], dtype='float32')
#     embeddings_index[word] = coefficients

# print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text data set')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label news_group to numeric id
labels = []  # list of label ids
for news_group in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, news_group)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[news_group] = label_id
        for post in sorted(os.listdir(path)):
            if post.isdigit():
                post_path = os.path.join(path, post)
                if sys.version_info < (3,):
                    f = open(post_path)
                else:
                    f = open(post_path, encoding='latin-1')
                texts.append(f.read())
                f.close()
                labels.append(label_id)

print('Found %s texts.' % len(texts))
print(texts[0])

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print('word index', word_index.values())

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Data tensor:', data)
print('Shape of label tensor:', labels.shape)
# print('Label tensor:', labels)

# # split the data into a training set and a validation set
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
# nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
#
# x_train = data[:-nb_validation_samples]
# y_train = labels[:-nb_validation_samples]
# x_val = data[-nb_validation_samples:]
# y_val = labels[-nb_validation_samples:]

x_test = data
y_test = labels

# print('Preparing embedding matrix.')
#
# # prepare embedding matrix
# nb_words = min(MAX_NB_WORDS, len(word_index))
# embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if i > MAX_NB_WORDS:
#         continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
#
# # load pre-trained word embeddings into an Embedding layer
# # trainable = False so as to keep the embeddings fixed
# embedding_layer = Embedding(nb_words + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=False)
#
# print('Training model.')
#
# # train a 1D convnet with global maxpooling
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(35)(x)
# x = Flatten()(x)
# x = Dense(128, activation='relu')(x)
# preds = Dense(len(labels_index), activation='softmax')(x)
#
# model = Model(sequence_input, preds)
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['acc'])
#
# model.fit(x_train, y_train, validation_data=(x_val, y_val),
#           nb_epoch=2, batch_size=128)
#
# #  Dec 30, 2016  loss: 0.3069 - acc: 0.8908 - val_loss: 0.1421 - val_acc: 0.9549
#
# # First evaluation of the model
# scores = model.evaluate(x_val, y_val, verbose=0)
# # scores = model.evaluate(x_val[0:test_partial, :], y_val[0:test_partial], verbose=0)
# print("Accuracy of Run Model: %.2f%%" % (scores[1]*100))
#
# model.save('../saved_models/0.0.3_model.h5')
#
# del model  # deletes the existing model

# returns a compiled model identical to the previous one
model = load_model('../saved_models/0.0.3_model.h5')
# print(model.summary())

# print('Predictions ...')
predictions = model.predict_on_batch(x_test)
print('Predictions: \n', predictions)
print('Labels: \n', labels)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy of Disk-Loaded Model: %.2f%%" % (scores[1]*100))

# Accuracy of Disk-Loaded Model: 11.11%
