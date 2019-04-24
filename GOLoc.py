import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())

# Parameters
BATCH_SIZE = 64
EPOCHS = 1000
learning_rate = 0.001

filename = ['../predictemall/parsedData.csv']

column_names = ['pokemonId', 'latitude', 'longitude', 'appearedHour', 'appearedMinute', 'temperature',
                'terrainType']

feature_names = column_names[1:]

output_name = column_names[0]

data = pd.read_csv("parsedData.csv", na_values="?", low_memory=False)
data.astype(np.float32)

# Randomly drop 75% of the data for ease of training
# data = data.sample(frac=0.25, random_state=0)
data = data.dropna()


# print(label_name)
# Create the output variables for what we want to predict
# y = data.pokemonId
# print(y.head)

data = tf.keras.utils.normalize(
    data,
    axis=-1,
    order=2
)

# Create training an test sets
# X = data.drop('pokemonId', axis=1)
# Split the data into test/train data 33% for validation 67% for training
train_dataset = data.sample(frac=0.67, random_state=0)
test_dataset = data.drop(train_dataset.index)
# print(train_dataset)

# Separate labels from the features
Y_train = train_dataset.pop('pokemonId')
Y_test = test_dataset.pop('pokemonId')


# Normalize the data
# X_train = tf.keras.utils.normalize(
#     train_dataset,
#     axis=-1,
#     order=2
# )
# X_test = tf.keras.utils.normalize(
#     test_dataset,
#     axis=-1,
#     order=2
# )
X_train = train_dataset
X_test = test_dataset


print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)

print("\ny_train:\n")
print(Y_train.head())
print(Y_train.shape)

print("\ny_test:\n")
print(Y_test.head())
print(Y_test.shape)


def build_model():
    # Final design
    model = Sequential()
    model.add(Dense(64, input_shape=[len(X_train.keys())], activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(64, activation='relu', kernel_initializer='random_normal'))
    model.add(Dropout(0.5))
    # model.add(Dense(56, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))

    # 8-12-24-48-24-12-1 Layer setup

    # model = Sequential()
    # model.add(Dense(8, input_shape=[len(X_train.keys())], activation='relu', kernel_initializer='random_normal'))
    # model.add(Dense(12, activation='relu', kernel_initializer='random_normal'))
    # model.add(Dropout(0.5))
    # model.add(Dense(24, activation='relu', kernel_initializer='random_normal'))
    # model.add(Dropout(0.5))
    # model.add(Dense(48, activation='relu', kernel_initializer='random_normal'))
    # model.add(Dropout(0.5))
    # model.add(Dense(24, activation='relu', kernel_initializer='random_normal'))
    # model.add(Dropout(0.5))
    # model.add(Dense(12, activation='relu', kernel_initializer='random_normal'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(1))

    # 5 64 node layer

    # model = Sequential()
    # model.add(Dense(64, input_shape=[len(X_train.keys())], activation='relu', kernel_initializer='random_normal'))
    # model.add(Dense(64, activation='relu', kernel_initializer='random_normal'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu', kernel_initializer='random_normal'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu', kernel_initializer='random_normal'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu', kernel_initializer='random_normal'))
    #
    # model.add(Dense(1))

    # Optimizer for Classification
    # sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    # Optimizer for Regression
    rms = tf.keras.optimizers.RMSprop(lr=learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=rms,
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    return model


# Regression model
def train_model(trn_model):
    # Fit the model
    mdl_hist = trn_model.fit(X_train, Y_train,
                             epochs=EPOCHS,
                             batch_size=BATCH_SIZE,
                             validation_data=(X_test, Y_test),
                             # validation_steps=int(176735 / BATCH_SIZE),
                             callbacks=[early_stop],
                             verbose=1)

    # evaluate the model
    loss, mae, mse = trn_model.evaluate(X_test, Y_test, verbose=0)
    print("Test Mean Absolute Error: {:5.2f} PokemonID" .format(mae))

    # Save model to JSON format
    model_json = trn_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # Save Weights
    trn_model.save_weights("model.h5")
    print("Model saved to folder.")

    return mdl_hist


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model.h5')
    print('Loaded model!')
    return loaded_model


def predict(loaded_model):

    test_predictions = loaded_model.predict(X_test).flatten()

    plt.scatter(Y_test, test_predictions)
    plt.xlabel('True Values [Pokemon]')
    plt.ylabel('Predictions [Pokemon]')
    plt.axis('equal')
    plt.axis('scaled')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-151, 151], [-151, 151])

    # error = test_predictions - Y_test
    # plt.hist(error, bins=25)
    # plt.xlabel("Prediction Error [MPG]")
    # _ = plt.ylabel("Count")

    plt.show()

    # test = X_train[:10]
    # # Predict the 10 highest numbers in the test data
    # prediction = model.predict(test)
    # print(prediction)
    # # top10 = [prediction[i] for i in prediction]


def plot_history(plt_history):
    # For Classification
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # x = range(1, len(acc) + 1)
    #
    # fig=plt.figure(figsize=(10, 5))
    # # plt.subplot(1, 2, 1)
    # # plt.plot(x, acc, 'b', label='Training acc')
    # # plt.plot(x, val_acc, 'r', label='Validation acc')
    # # plt.title('Training and validation accuracy')
    # # plt.legend()
    # plt.subplot(1, 1, 1)
    # plt.plot(x, loss, 'b', label='Training loss')
    # plt.plot(x, val_loss, 'r', label='Validation loss')
    # plt.title('Training and Validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # For Regression
    hist = pd.DataFrame(plt_history.history)
    hist['epoch'] = plt_history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Pokemon]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    # plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [Pokemon]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    # plt.ylim([0, 20])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    build_model = build_model()
    # This will stop the model if it detects no improvement over time
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = train_model(build_model)

    plot_history(history)

    predict_model = load_model()
    predict(predict_model)

    # print("Predictions:\n")
    # top10 = predict(model)
    # print(top10)
