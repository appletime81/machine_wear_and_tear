from tensorflow.keras.datasets import boston_housing


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


print(x_train.shape)
print(y_train.shape)