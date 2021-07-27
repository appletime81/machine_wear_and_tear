"""
this program is using Karas's example to modify.
Reference: https://iter01.com/487192.html
"""
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model #泛型模型
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

## YC's data preparation
import numpy as np
import pandas as pd

filepath="4parameters.xlsx"
all_df=pd.read_excel(filepath)
all_df[:3]
cols=['worn','platform RMS','platform rsf2','platform fft total']
all_df=all_df[cols]
all_df['worn']=all_df['worn']/5
all_df[:3]

# train and test rate
msk=np.random.rand(len(all_df))<0.7
x_train=all_df[msk] #70% data train
x_test=all_df[~msk] #30% data test
print('total:',len(all_df),'train:',len(x_train),'test:',len(x_test))
print(all_df)


# preprocess data function
def PreprocessData(raw_df):
    df=raw_df
    x_onehot_df=pd.get_dummies(data=df,columns=['worn'])
    ndarray=x_onehot_df.values
    Features=ndarray[:,1:3]
    Label=ndarray[:,0]
    return Features, Label

# # X shape (60,000 28x28), y shape (10,000, )
# (x_train, _), (x_test, y_test) = mnist.load_data()

# # 資料預處理
# x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
# x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
# x_train = x_train.reshape((x_train.shape[0], -1))
# x_test = x_test.reshape((x_test.shape[0], -1))
# print(x_train.shape)
# print(x_test.shape)

# 壓縮特徵維度至2維
encoding_dim = 2

# this is our input placeholder
input_img = Input(shape=(784,))

# 編碼層
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# 解碼層
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

# 構建自編碼模型
autoencoder = Model(inputs=input_img, outputs=decoded)

# 構建編碼模型
encoder = Model(inputs=input_img, outputs=encoder_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True)

# plotting
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, s=3)
plt.colorbar()
plt.show()