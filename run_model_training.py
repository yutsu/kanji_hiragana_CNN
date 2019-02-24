from preprocessing.get_label_set import get_label_txt
from preprocessing.make_input_data import input_data
from keras_model import keras_model
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau


epochs = 10
batch_size = 86
lr = 0.001


def load_model_weights(name, model):
    try:
        model.load_weights(name)
        print('mode loaded')
    except:
        print("Can't load weights!")


def save_model_weights(name, model):
    try:
        model.save_weights(name)
        print("saved classifier weights")
    except:
        print("failed to save classifier weights")
    pass


X_train, Y_train, X_test, Y_test, unique_labels = input_data()
n_output = Y_train.shape[1]


# If you haven't created labels.txt yet
# get_label_txt(unique_labels)


model = keras_model(n_output)
# model.summary()


load_model_weights('weights/keras_weights-2.h5', model)
optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


training = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test), verbose=2)

save_model_weights('weights/keras_weights-2.h5', model)
model.save('keras.h5')
