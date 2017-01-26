import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import sys
from keras.models import load_model
import pickle

np.random.seed(1337)

BASE_DIR = '/Users/kimardenmiller/local_data/NLP_Embeddings/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
VALIDATION_SPLIT = 0.2

print('Processing text data set ...')

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

tokenizer = pickle.load(open('../saved_models/tokenizer.004.p', 'rb'))
sequences = tokenizer.texts_to_sequences(texts)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
# print('Shape of data tensor:', data.shape)
# print('Data tensor:', data)
# print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
# nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
#
# # x_train = data[:-nb_validation_samples]
# # y_train = labels[:-nb_validation_samples]
# x_val = data[-nb_validation_samples:]
# y_val = labels[-nb_validation_samples:]

x_test = data
y_test = labels

# returns a compiled model identical to the previous one
model = load_model('../saved_models/0.0.4_model.h5')
# print(model.summary())

# print('Predictions ...')
# predictions = model.predict_on_batch(x_val)
# print('Predictions: \n', predictions)
# print('Labels tensor: \n', labels)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy of Disk-Loaded Model: %.2f%%" % (scores[1]*100))

