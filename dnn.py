import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from glob import glob
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from main import *


def data_preprocessing(data):
    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data, scaler


def dnn_model(x_train):
  model = Sequential()
  model.add(Dense(32, input_shape=(x_train.shape[1],), activation="relu"))
  model.add(Dense(64, activation="relu"))
  model.add(Dense(512, activation="relu"))
  model.add(Dense(3, activation="softmax"))
  return model


def generate_result(data):
    encoding_dim = 1000
    input_img = Input(shape=(20000,))

    # 編碼層
    encoded = Dense(10000, activation='relu')(input_img)
    encoded = Dense(5000, activation='relu')(encoded)
    encoded = Dense(2000, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # 解碼層
    decoded = Dense(2000, activation='relu')(encoder_output)
    decoded = Dense(5000, activation='relu')(decoded)
    decoded = Dense(10000, activation='relu')(decoded)
    decoded = Dense(20000, activation='tanh')(decoded)

    # 構建自編碼模型
    # autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoder_output)
    encoder.load_weights("encoder.h5")
    result = encoder.predict(data)
    return result


if __name__ == "__main__":
    # ------------------------------declare file name variables----------------------------------------
    new_data_folder = "8000rpm_new_csv"
    final_data_folder = "8000rpm_final_csv"
    wear_data_folder = "8000rpm_wear_csv"

    # --------------------------------------generate data----------------------------------------------
    new_data = concat_data(generate_data_list(load_data(new_data_folder)))
    final_data = concat_data(generate_data_list(load_data(final_data_folder)))
    wear_data = concat_data(generate_data_list(load_data(wear_data_folder)))

    x_train = np.concatenate((new_data[0:int(10 * 0.8), :],
                              final_data[0:int(10 * 0.8), :],
                              wear_data[0:int(10 * 0.8), :]), axis=0)
    x_test = np.concatenate((new_data[int(10 * 0.8):, :],
                             final_data[int(10 * 0.8):, :],
                             wear_data[int(10 * 0.8):, :]), axis=0)

    x_data = np.concatenate((x_train, x_test), axis=0)
    x_data_after_scale, x_scaler = min_max_scale(x_data)

    x_train_after_scale = x_data_after_scale[:24, :]
    x_test_after_scale = x_data_after_scale[24:, :]
    print(x_train_after_scale.shape)

    y_train, y_test = generate_label()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_label = np.concatenate((y_train, y_test), axis=0)
    print(y_test)

    x_new_train = generate_result(x_train_after_scale)
    x_new_test = generate_result(x_test_after_scale)
    x_new_total = generate_result(x_data_after_scale)

    early_stop = EarlyStopping(monitor='val_loss', patience=15)

    model = dnn_model(x_new_train)
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    # history = model.fit(x_new_train, y_train, epochs=100, batch_size=8, verbose=1, validation_data=(x_new_test, y_test))
    history = model.fit(x_new_total, y_label, epochs=100, batch_size=8, verbose=1, validation_split=0.1)
    model.save("dnn_wear_and_tear_005.h5")

    plt.figure(figsize=(15, 8))
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(history.epoch, history.history['loss'], label='Train Loss')
    plt.plot(history.epoch, history.history['val_loss'], label='Val loss')
    plt.title('Model loss')
    plt.legend()
    plt.show()



