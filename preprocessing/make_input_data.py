import numpy as np

from keras.utils import np_utils
from preprocessing.data_utils import process_ETL_data
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def input_data(test_size=0.2):
    size = (64, 64)

    for i in range(1, 4):
        if i == 3:
            max_records = 315
        else:
            max_records = 319

        if i != 1:
            start_record = 0
        else:
            start_record = 75

        chars, labs = process_ETL_data(i, range(start_record, max_records))

        if i == 1:
            characters = chars
            labels = labs

        else:
            characters = np.concatenate((characters, chars), axis=0)
            labels = np.concatenate((labels, labs), axis=0)

    chars, labs = process_ETL_data(1, range(0, 75))

    characters = np.concatenate((characters, chars), axis=0)
    labels = np.concatenate((labels, labs), axis=0)

    # rename labels from 0 to 952 (953 Japanese characters (hiragana, kanji))
    unique_labels = list(set(labels))
    labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
    new_labels = np.array([labels_dict[l] for l in labels], dtype=np.int32)

    characters_shuffle, new_labels_shuffle = shuffle(characters, new_labels, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(characters_shuffle, new_labels_shuffle, test_size=test_size, random_state=0)

    # reshape to (64, 64, 1)
    X_train = x_train.reshape(
        (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    X_test = x_test.reshape(
        (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    # onehot encoding
    n_classes = len(unique_labels)
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)

    return X_train, Y_train, X_test, Y_test, unique_labels
