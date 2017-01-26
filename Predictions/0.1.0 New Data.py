import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import sys
from keras.models import load_model
from keras.utils.np_utils import probas_to_classes
import pickle

np.random.seed(1337)

BASE_DIR = '/Users/kimardenmiller/Dropbox/PyCharm/NLP_CNN/Prediction_New_Inputs'
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

target_start = 0
target_stop = 13
target_predictions = np.arange(target_start, target_stop)

print('Found %s texts.' % len(texts))
for target in target_predictions:
    print('Target blog: ', target, '\n', texts[target][0:1000], '\n __________ \n')

tokenizer = pickle.load(open('../saved_models/tokenizer.004.p', 'rb'))
sequences = tokenizer.texts_to_sequences(texts)

x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y_test = to_categorical(np.asarray(labels))
# print('Shape of data tensor:', x_test.shape)
# print('Data tensor:', x_test)
# print('Shape of label tensor:', labels.shape)

# returns a compiled model identical to the previous one
model = load_model('../saved_models/0.0.4_model.h5')
# print(model.summary())

print('Generating Predictions ...\n')
print('Sample ', target_stop - target_start, 'Predictions: \n')
predictions = model.predict_on_batch(x_test)
class_predictions = probas_to_classes(predictions)

# print('Predictions: \n', predictions[target_start:target_stop])
# print('Class Prediction: \n', class_predictions[target_start:target_stop])
# print('Labels tensor: \n', cat_labels[target_start:target_stop])
# print('labels index: ', labels_index.items())
for target in target_predictions:
    print('Correct Category:   ', [label for (label, v) in labels_index.items() if v == labels[target]][0])
    print('Predicted Category: ', [label for (label, v) in labels_index.items() if v == class_predictions[target]][0], '\n')

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy of Disk-Loaded Model: %.2f%%" % (scores[1]*100))

# Sample  13 Predictions below:
#
# Correct Category:    alt.atheism
# Predicted Category:  soc.religion.christian
#
# Correct Category:    alt.atheism
# Predicted Category:  soc.religion.christian
#
# Correct Category:    alt.atheism
# Predicted Category:  alt.atheism
#
# Correct Category:    comp.graphics
# Predicted Category:  comp.graphics
#
# Correct Category:    comp.graphics
# Predicted Category:  comp.sys.mac.hardware
#
# Correct Category:    comp.graphics
# Predicted Category:  comp.graphics
#
# Correct Category:    comp.os.ms-windows.misc
# Predicted Category:  comp.os.ms-windows.misc
#
# Correct Category:    comp.os.ms-windows.misc
# Predicted Category:  comp.os.ms-windows.misc
#
# Correct Category:    comp.os.ms-windows.misc
# Predicted Category:  comp.os.ms-windows.misc
#
# Correct Category:    comp.sys.ibm.pc.hardware
# Predicted Category:  comp.sys.ibm.pc.hardware
#
# Correct Category:    comp.sys.ibm.pc.hardware
# Predicted Category:  comp.sys.ibm.pc.hardware
#
# Correct Category:    comp.sys.ibm.pc.hardware
# Predicted Category:  comp.sys.ibm.pc.hardware
#
# Correct Category:    comp.sys.mac.hardware
# Predicted Category:  comp.sys.ibm.pc.hardware
#
# Accuracy of Disk-Loaded Model: 82.14%