# Сверточная нейронная сеть
# 0.8 -> 0.5820177921010591 0.5857558512649454 0.5838760533453207 0.5837181494327802 0.5834722998070162
# 0.8 -> 0.5888496426925353 0.58388413397635 0.5902611350268783 0.5904798420889166 0.5854869997322765
# 0.9 -> 0.6020819156585703 0.604149407892614 0.6055090115120009 0.6134101531380093 0.6059625127749051
# 0.9 -> 0.6052020879997299 0.6061862462466698 0.6076906502348954 0.6056515337988616 0.5981331411444261
import csv
import numpy as np
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers.convolutional import Conv1D
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
# Изменение размерности данных, т.к. в CNN принимаются трехмерные массивы
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
# Создание слоев сети
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=20, activation='relu', input_shape=(43, 1)))
model.add(Flatten())
model.add(Dense(4, activation='sigmoid'))
opt = Adam(lr=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
# Компилирование сети
model.compile(optimizer=opt,
              loss='mse',
              metrics=['accuracy'])
# Обучение сети
history = model.fit(X_train, Y_train,
          epochs=250,
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