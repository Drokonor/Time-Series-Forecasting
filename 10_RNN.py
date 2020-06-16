# Рекурентная нейронная сеть
# 0.8 -> 0.5814112244005137 0.577261871303583 0.5818382633812049 0.582466871495671 0.5658737540781131
# 0.8 -> 0.5706424721315443 0.5856901972715256 0.571958005521587 0.5841375128300318 0.5682027739507438
# 0.9 -> 0.6062849490974751 0.6007939060583303 0.5962615951240735 0.6030428480061971 0.606164235102127
# 0.9 -> 0.6036810762538904 0.6068683830308073 0.5998042887442206 0.60368154958511 0.6021847189838277
import csv
import numpy as np
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
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
# Изменение размерности данных, т.к. в LSTM принимаются трехмерные массивы
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
# Создание слоев сети
model = Sequential()
model.add(LSTM(128, input_shape=(43, 1), return_sequences=True))
model.add(LeakyReLU())
model.add(LSTM(64))
model.add(LeakyReLU())
model.add(Dense(4, activation='sigmoid'))
opt = Adam(lr=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
# Компилирование сети
model.compile(optimizer=opt,
              loss='mse',
              metrics=['accuracy'])
# Обучение сети
history = model.fit(X_train, Y_train,
          epochs=80,
          batch_size=256,
          verbose=2,
          validation_data=(X_train, Y_train),
          shuffle=True,
          callbacks=[reduce_lr]
          )
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