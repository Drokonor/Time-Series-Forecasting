# Интегрированная модель авторегрессии — скользящего среднего
# 1,1,0 -> 0.55865737
# 2,1,0 -> 0.56346747
# 3,1,0 -> 0.56347504
# 0,1,1 -> 0.58136999
# 0,2,1 -> 0.53990406
# 1,1,1 -> 0.57786915
# 1,2,1 -> 0.55212327
# 2,1,1 -> 0.57295555
# 3,1,1 -> 0.56135009
# 1,2,1 -> 0.55212327
# 2,2,1 -> 0.55243548
# 0,1,2 -> 0.58011033
# 1,1,2 -> 0.56653403
# 2,1,2 -> 0.55825889
# 3,1,2 -> 0.55052859
# 0,1,3 -> 0.58311852
# 1,1,3 -> 0.54807645
# 2,1,3 -> 0.54176875
# 3,1,3 -> 0.5264616
from statsmodels.tsa.statespace.sarimax import SARIMAX
import csv
import numpy
import matplotlib.pylab as plt
True_values = []
Predicted_values = []
Acc_list = []
# Выбор параметров p,d и q
p = 3
d = 1
q = 3
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
                model = SARIMAX(data[0:len(data) - i - 1], order=(p, d, q))
                model_fit = model.fit(disp=False)
                # Прогноз
                prediction = model_fit.predict(len(data[0:len(data) - i - 1]),
                                               len(data[0:len(data) - i - 1]), typ='levels')
                # Модификация для интервала [0;1]
                if prediction[0] > 1:
                    prediction[0] = float(1)
                if prediction[0] < 0:
                    prediction[0] = float(0)
                True_values.append(data[len(data) - i - 1])
                Predicted_values.append(prediction[0])
                # Вычисление точности прогноза
                if float(data[len(data) - i - 1]) == 0 and prediction == 0:
                    acc = 1
                else:
                    acc = min(data[len(data) - i - 1], prediction) / max(data[len(data) - i - 1], prediction)
                Acc_list.append(acc)
                # Изменение значения наблюдения на прогнозируемое для дальнейшего прогноза следующих наблюдений
                data[len(data) - i - 1] = prediction[0]
            # При ошибках в прогнозировании точность становится равной нулю
            except ZeroDivisionError:
                Acc_list.append(0)
                continue
            except numpy.linalg.LinAlgError:
                Acc_list.append(0)
                continue
            except IndexError:
                Acc_list.append(0)
                continue
# Вычисление средней точности прогноза
Average_Acc = sum(Acc_list)/len(Acc_list)
print(Average_Acc)
# График фактических и прогнозируемых значений наблюдений
plt.plot(True_values, color='black', label = 'Original data')
plt.plot(Predicted_values, color='blue', label = 'Predicted data')
plt.legend(loc='best')
plt.title('Actual and predicted')
plt.show()