# Модель экспоненциального сглаживания
# 0.26 0.5968262746700852
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import csv
import matplotlib.pylab as plt
All_acc_list = []
# Цикл для проверки всех значений коэффициента сглаживания от 0 до 1 с шагом 0.01
for S in range(101):
    Smooth = S/100
    Acc_list = []
    for num in range(1146):
        # Считывание входных данных
        with open('Time_Series/item' + str(1213 + num) + '.csv') as f:
            reader = csv.reader(f)
            month_quantity = 0
            for row in reader:
                month_quantity += 1
        # Исключение рядов длиной меньше 5
        if month_quantity > 4:
            with open('Time_Series/item' + str(1213 + num) + '.csv') as f:
                reader = csv.reader(f)
                data = []
                counter = 0
                for row in reader:
                    if 0 < counter <= month_quantity - 1:
                        data.append(float(row[2]))
                    counter += 1
            # Цикл с 4 итерациями для прогнозирования 4 значений
            for i in [3, 2, 1, 0]:
                try:
                    # Создание модели
                    model = SimpleExpSmoothing(data[0:len(data) - i - 1])
                    model_fit = model.fit(smoothing_level=Smooth)
                    # Прогноз
                    prediction = model_fit.predict(len(data[0:len(data) - i - 1]), len(data[0:len(data) - i - 1]))
                    # Модификация для интервала [0;1]
                    if prediction[0] > 1:
                        prediction[0] = float(1)
                    if prediction[0] < 0:
                        prediction[0] = float(0)
                    # Вычисление точности прогноза
                    if float(data[len(data) - i - 1]) == 0 and prediction == 0:
                        acc = 1
                    else:
                        acc = min(data[len(data) - i - 1], prediction) / max(data[len(data) - i - 1], prediction)
                    Acc_list.append(acc)
                    # Изменение значения наблюдения на прогнозируемое для дальнейшего прогноза следующих наблюдений
                    data[len(data) - i - 1] = prediction[0]
                # При ошибках в прогнозировании точность становится равной нулю
                except IndexError:
                    Acc_list.append(0)
                    continue
    # Вычисление средней точности прогноза
    Average_Acc = sum(Acc_list)/len(Acc_list)
    All_acc_list.append(Average_Acc)
# Нахождение максимальной средней точности и коэффициента сглаживания дающего эту точность
M = max(All_acc_list)
for i in range (len(All_acc_list)):
    if All_acc_list[i] == M:
        best = i
print(M)
print(best)
# График зависимости средней точности от коэффициента сглаживания
plt.plot(All_acc_list, color='blue', label = 'Accuracy')
plt.legend(loc='best')
plt.title('Average accuracy for different smoothing level')
plt.show()