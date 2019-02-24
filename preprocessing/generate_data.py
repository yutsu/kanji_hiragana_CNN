"""Generate an example dataset for 160 different writers."""
"""https://github.com/charlietsai/japanese-handwriting-nn/blob/master/preprocessing/generate_data.py"""


# import cPickle as pickle
import _pickle as pickle
import numpy as np
from data_utils import get_ETL_data

writersPerChar = 160

for i in range(1, 4):
    if i == 3:
        max_records = 315
    else:
        max_records = 319

    chars, labs = get_ETL_data(
        i, range(0, max_records), writersPerChar, vectorize=True, resize=(28, 28))
    if i == 1:
        characters = chars
        labels = labs
    characters = np.concatenate((characters, chars), axis=0)
    labels = np.concatenate((labels, labs), axis=0)

print(characters)

# rename labels from 0 to n_labels-1
unique_labels = list(set(labels))
labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
new_labels = np.array([labels_dict[l] for l in labels], dtype=np.int32)

pickle.dump((characters, new_labels), open('160_writers.pckl', 'wb'))
