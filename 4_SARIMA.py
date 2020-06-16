# Сезонная интегрированная модель авторегрессии — скользящего среднего
# 2 -> 0.5977918060459139
# 3 -> 0.5953569169058407
# 4 -> 0.5875246459866841
# 5 -> 0.5848328648602573
# 6 -> 0.5865294289538442
from statsmodels.tsa.statespace.sarimax import SARIMAX
import csv
import numpy
import matplotlib.pylab as plt
True_values = []
Predicted_values = []
Acc_list = []
for num in range(1146):
    print(num)
    with open('Time_Series/item' + str(1213 + num) + '.csv') as f:
        reader = csv.reader(f)
        month_quantity = 0
        for row in reader:
            month_quantity += 1
    with open('Time_Series/item' + str(1213 + num) + '.csv') as f:
        reader = csv.reader(f)
        data = []
        counter = 0
        for row in reader:
            if 0 < counter <= month_quantity - 1:
                data.append(float(row[2]))
            counter += 1
    if len(data) > 5:
        try:
            for i in [3, 2, 1, 0]:
                model = SARIMAX(data[0:len(data) - i - 1], order=(1, 1, 1), seasonal_order=(1, 1, 1, 2))
                model_fit = model.fit(disp=False)
                prediction = model_fit.predict(len(data[0:len(data) - i - 1]), len(data[0:len(data) - i - 1]))
                if prediction[0] > 1:
                    prediction[0] = float(1)
                if prediction[0] < 0:
                    prediction[0] = float(0)
                True_values.append(data[len(data) - i - 1])
                Predicted_values.append(prediction[0])
                if float(data[len(data) - i - 1]) != 0 and prediction[0] != 0:
                    acc = min(data[len(data) - i - 1], prediction[0])/max(data[len(data) - i - 1], prediction[0])
                else:
                    acc = 1
                Acc_list.append(acc)
                data[len(data) - i - 1] = prediction[0]
                del model
                del model_fit
        except numpy.linalg.LinAlgError:
            continue
        except IndexError:
            continue
Average_Acc = sum(Acc_list)/len(Acc_list)
print(Average_Acc)
plt.plot(True_values, color='black', label = 'Original data')
plt.plot(Predicted_values, color='blue', label = 'Predicted data')
plt.legend(loc='best')
plt.title('Actual and predicted')
plt.show()