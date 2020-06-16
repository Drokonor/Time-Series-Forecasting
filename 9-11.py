import csv
import numpy as np
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Flatten
from keras.layers.advanced_activations import *
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers.convolutional import Conv1D
data = []
for num in range(1146):
    with open('Time_Series/item' + str(1213 + num) + '.csv') as f:
        reader = csv.reader(f)
        month_quantity = 0
        for row in reader:
            month_quantity += 1
    with open('Time_Series/item' + str(1213 + num) + '.csv') as f:
        reader = csv.reader(f)
        one_data = []
        counter = 0
        for row in reader:
            if 0 < counter <= month_quantity - 1:
                one_data.append(float(row[2]))
            counter += 1
        if len(one_data) > 4:
            data.append(one_data)
train, test = ((data[0:int(len(data) * 0.9)], data[int(len(data) * 0.9) + 1:]))
for i in range(len(train)):
    for j in range(len(train[i]), 47):
        train[i].insert(0, 0)
for i in range(len(test)):
    for j in range(len(test[i]), 47):
        test[i].insert(0, 0)
train = np.array(train)
test = np.array(test)
X_train = train[:, 0:-4]
X_test = test[:, 0:-4]
Y_train = train[:, -4:]
Y_test = test[:, -4:]
model = Sequential()
model.add(Dense(64, input_dim=43))
model.add(LeakyReLU())
model.add(Dense(32))
model.add(LeakyReLU())
model.add(Dense(16))
model.add(LeakyReLU())
model.add(Dense(4))
model.add(Activation('sigmoid'))
opt = Adam(lr=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
model.compile(optimizer=opt,
              loss='mse',
              metrics=['accuracy'])
#model.load_weights('NN_model_08')
history = model.fit(X_train, Y_train,
          epochs=100,
          batch_size=512,
          verbose=2,
          validation_data=(X_train, Y_train),
          shuffle=True,
          callbacks=[reduce_lr])
Predicted_values_1 = model.predict(np.array(X_test))
True_values = Y_test
Predicted_values_1 = Predicted_values_1.reshape(Predicted_values_1.shape[0] * Predicted_values_1.shape[1])
True_values = True_values.reshape(True_values.shape[0]*True_values.shape[1])
#model.save_weights('NN_model_08')
Y_test[Y_test == 0] = 1e-10
Predicted_values_1[Predicted_values_1 == 0] = 1e-10
tmp = True_values / Predicted_values_1
tmp[tmp > 1] = 1/tmp[tmp > 1]
print(np.sum(tmp) / (Predicted_values_1.shape))
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
model = Sequential()
model.add(LSTM(128, input_shape=(43, 1), return_sequences=True))
model.add(LeakyReLU())
model.add(LSTM(64))
model.add(LeakyReLU())
model.add(Dense(4, activation='sigmoid'))
opt = Adam(lr=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
model.compile(optimizer=opt,
              loss='mse',
              metrics=['accuracy'])
#model.load_weights('NN_model_08')
history = model.fit(X_train, Y_train,
          epochs=80,
          batch_size=256,
          verbose=2,
          validation_data=(X_train, Y_train),
          shuffle=True,
          callbacks=[reduce_lr]
          )
Predicted_values_2 = model.predict(np.array(X_test))
True_values = Y_test
Predicted_values_2 = Predicted_values_2.reshape(Predicted_values_2.shape[0] * Predicted_values_2.shape[1])
True_values = True_values.reshape(True_values.shape[0] * True_values.shape[1])
#model.save_weights('NN_model_08')
Y_test[Y_test == 0] = 1e-10
Predicted_values_2[Predicted_values_2 == 0] = 1e-10
tmp = True_values / Predicted_values_2
tmp[tmp > 1] = 1/tmp[tmp > 1]
print(np.sum(tmp) / (Predicted_values_2.shape))
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=20, activation='relu', input_shape=(43, 1)))
model.add(Flatten())
model.add(Dense(4, activation='sigmoid'))
opt = Adam(lr=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
model.compile(optimizer=opt,
              loss='mse',
              metrics=['accuracy'])
#model.load_weights('CNN_model_08')
history = model.fit(X_train, Y_train,
          epochs=250,
          batch_size=256,
          verbose=2,
          validation_data=(X_train, Y_train),
          shuffle=True,
          callbacks=[reduce_lr]
          )
Predicted_values_3 = model.predict(np.array(X_test))
True_values = Y_test
Predicted_values_3 = Predicted_values_3.reshape(Predicted_values_3.shape[0] * Predicted_values_3.shape[1])
True_values = True_values.reshape(True_values.shape[0] * True_values.shape[1])
#model.save_weights('CNN_model_08')
Y_test[Y_test == 0] = 1e-10
Predicted_values_3[Predicted_values_3 == 0] = 1e-10
tmp = True_values / Predicted_values_3
tmp[tmp > 1] = 1/tmp[tmp > 1]
print(np.sum(tmp) / (Predicted_values_3.shape))
plt.plot(True_values[0:100], color='black', label = 'Original data')
plt.plot(Predicted_values_1[0:100], color='blue', label = 'MLP')
plt.plot(Predicted_values_2[0:100], color='green', label = 'LSTM')
plt.plot(Predicted_values_3[0:100], color='yellow', label = 'CNN')
plt.legend(loc='best')
plt.title('Actual and predicted 9:1')
plt.show()