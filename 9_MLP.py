# Нейронная сеть
# 0.8 -> 0.5802234184939641 0.5892402058654964 0.5677699269366836 0.570430443075058 0.5706445809914853
# 0.8 -> 0.574515104316251 0.5687016764406692 0.5659476291466247 0.574392420822243 0.5696603260337368
# 0.9 -> 0.5905625682291824 0.5862358678514141 0.6030914074158763 0.5910498309335426 0.5895571298434727
# 0.9 -> 0.5901402460295648 0.5850174120046018 0.5871832992080448 0.5868608482857637 0.5920629801272034
import csv
import numpy as np
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import *
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
data = []
# Считывание всех входных данных
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
        # Исключение рядов длиной меньше 5
        if len(one_data) > 4:
            data.append(one_data)
# Разбиение на выборки
train, test = ((data[0:int(len(data) * 0.8)], data[int(len(data) * 0.8) + 1:]))
# Заполнение рядов нулями в их начале для того, чтобы входной слой сети был полностью заполнен
for i in range(len(train)):
    for j in range(len(train[i]), 47):
        train[i].insert(0, 0)
for i in range(len(test)):
    for j in range(len(test[i]), 47):
        test[i].insert(0, 0)
train = np.array(train)
test = np.array(test)
# Отведение 4 последних наблюдений на прогнозирование
X_train = train[:, 0:-4]
X_test = test[:, 0:-4]
Y_train = train[:, -4:]
Y_test = test[:, -4:]
# Создание слоев сети
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
# Компилирование сети
model.compile(optimizer=opt,
              loss='mse',
              metrics=['accuracy'])
# Обучение сети
history = model.fit(X_train, Y_train,
          epochs=100,
          batch_size=512,
          verbose=2,
          validation_data=(X_train, Y_train),
          shuffle=True,
          callbacks=[reduce_lr])
# График фактических и прогнозируемых значений наблюдений
Predicted_values = model.predict(np.array(X_test))
True_values = Y_test
Predicted_values = Predicted_values.reshape(Predicted_values.shape[0] * Predicted_values.shape[1])
True_values = True_values.reshape(True_values.shape[0] * True_values.shape[1])
plt.plot(True_values, color='black', label = 'Original data')
plt.plot(Predicted_values, color='blue', label = 'Predicted data')
plt.legend(loc='best')
plt.title('Actual and predicted')
plt.show()
# Вычисление средней точности прогноза
True_values[True_values == 0] = 1e-10
Predicted_values[Predicted_values == 0] = 1e-10
tmp = True_values/Predicted_values
tmp[tmp > 1] = 1/tmp[tmp > 1]
print(np.sum(tmp)/(Predicted_values.shape))
