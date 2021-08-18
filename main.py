import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler


def load_data(folder):
    return glob(f"{folder}/*.csv")


def generate_data_list(file_paths):
    df_list = []
    for file_path in file_paths:
        temp_data = pd.read_csv(file_path).to_numpy()
        temp_data = temp_data.reshape(1, -1)
        df_list.append(temp_data)
    return df_list


def concat_data(df_list):
    result = np.concatenate(tuple(df_list), axis=0)
    return result


def autoencoder_model(x_train, x_test):
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
    autoencoder = Model(inputs=input_img, outputs=decoded)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    history = autoencoder.fit(x_train, x_train, epochs=20, batch_size=2, verbose=1, validation_data=(x_test, x_test))

    # plotting
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig('training_loss4.png')
    plt.show()

    # 構建編碼模型
    encoder = Model(inputs=input_img, outputs=encoder_output)
    encoder.save('encoder3.h5')


def generate_label():
    new_data_label_list = []
    final_data_label_list = []
    wear_data_label_list = []

    for i in range(10):
        new_data_label_list.append(0)

    for i in range(10):
        final_data_label_list.append(1)

    for i in range(10):
        wear_data_label_list.append(2)

    new_data_label = np.array(new_data_label_list)
    final_data_label = np.array(final_data_label_list)
    wear_data_label = np.array(wear_data_label_list)

    y_train = np.concatenate(
        (new_data_label[:int(10 * 0.8)], final_data_label[:int(10 * 0.8)], wear_data_label[:int(10 * 0.8)]),
        axis=0)
    y_test = np.concatenate(
        (new_data_label[int(10 * 0.8):], final_data_label[int(10 * 0.8):], wear_data_label[int(10 * 0.8):]),
        axis=0)
    return y_train, y_test


def min_max_scale(data):
    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data, scaler


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
    print(x_train_after_scale.shape)
    x_test_after_scale = x_data_after_scale[24:, :]
    print(x_test_after_scale.shape)

    # -------------------------------------------------training----------------------------------------
    autoencoder_model(x_train_after_scale, x_test_after_scale)


