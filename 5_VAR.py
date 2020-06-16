# Модель векторной авторегрессии
# 2,1 -> 0.5510586561779971
# 2,2 -> 0.5501869157548623
# 2,3 -> 0.5502120350927303
# 2,4 -> 0.5677788790972621
# 3,1 -> 0.5649594310410941
# 3,2 -> 0.561369889138184
# 3,3 -> 0.5603552959638872
# 3,4 -> 0.5457583645712256
# 4,1 -> 0.5692923384776719
# 4,2 -> 0.5655375687716939
# 4,3 -> 0.5633357928431163
# 4,4 -> 0.5484822883719773
# 5,1 -> 0.5712443838276741
# 5,2 -> 0.5674650079689864
# 5,3 -> 0.5649765334439882
# 5,4 -> 0.5503669166140569
# 6,1 -> 0.5672349544297273
# 6,2 -> 0.5568444678559437
# 6,3 -> 0.5565987499303702
# 6,4 -> 0.5451280809436525
from statsmodels.tsa.vector_ar.var_model import VAR
import csv
import matplotlib.pylab as plt
True_values = []
Predicted_values = []
Acc_list = []
# Выбор параметра p
p = 1
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
                    x = []
                    for i in range(2):
                        x.append(float(row[2]))
                    data.append(x)
                counter += 1
        # Цикл с 4 итерациями для прогнозирования 4 значений
        for i in [3, 2, 1, 0]:
            try:
                # Создание модели
                model = VAR(data[0:len(data) - i - 1])
                model_fit = model.fit(maxlags=p)
                # Прогноз
                prediction = model_fit.forecast(model_fit.y, steps=1)
                # Модификация для интервала [0;1]
                if prediction[0][0] > 1:
                    prediction[0][0] = float(1)
                if prediction[0][0] < 0:
                    prediction[0][0] = float(0)
                True_values.append(data[len(data) - i - 1][0])
                Predicted_values.append(prediction[0][0])
                # Вычисление точности прогноза
                if float(data[len(data) - i - 1][0]) == 0 and prediction[0][0] == 0:
                    acc = 1
                else:
                    acc = min(data[len(data) - i - 1][0], prediction[0][0]) / max(data[len(data) - i - 1][0],
                                                                                  prediction[0][0])
                Acc_list.append(acc)
                # Изменение значения наблюдения на прогнозируемое для дальнейшего прогноза следующих наблюдений
                for j in range(2):
                    data[len(data) - i - 1][j] = prediction[0][0]
            # При ошибках в прогнозировании точность становится равной нулю
            except ValueError:
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