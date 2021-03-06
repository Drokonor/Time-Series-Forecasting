# Модель авторегрессионной условной гетероскедастичности
# 1 -> 0.34416571763919557
# 2 -> 0.34213932688386883
# 3 -> 0.3413337743061046
# 4 -> 0.34162381059964886
# 5 -> 0.3408732746997321
from arch import arch_model
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
                    data.append(float(row[2]))
                counter += 1
        # Цикл с 4 итерациями для прогнозирования 4 значений
        for i in [3, 2, 1, 0]:
            try:
                # Создание модели
                model = arch_model(data[0:len(data) - i - 1], mean='Zero', vol='ARCH', p=p, rescale=True)
                model_fit = model.fit()
                # Прогноз
                prediction_0 = model_fit.forecast(horizon=1)
                prediction = prediction_0.variance.values[len(prediction_0.variance.values) - 1][0]
                # Модификация для интервала [0;1]
                if prediction > 1:
                    prediction = float(1)
                if prediction < 0:
                    prediction = float(0)
                True_values.append(data[len(data) - i - 1])
                Predicted_values.append(prediction)
                # Вычисление точности прогноза
                if float(data[len(data) - i - 1]) == 0 and prediction == 0:
                    acc = 1
                else:
                    acc = min(data[len(data) - i - 1], prediction) / max(data[len(data) - i - 1], prediction)
                Acc_list.append(acc)
                # Изменение значения наблюдения на прогнозируемое для дальнейшего прогноза следующих наблюдений
                data[len(data) - i - 1] = prediction
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