# Наивная модель
# 0.5723293745399595
import csv
import matplotlib.pylab as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA
import numpy
True_values = []
Predicted_values_1 = []
Acc_list = []
for num in range(1146):
    with open('Time_Series/item' + str(1213 + num) + '.csv') as f:
        reader = csv.reader(f)
        month_quantity = 0
        for row in reader:
            month_quantity += 1
    if month_quantity > 4:
        with open('Time_Series/item' + str(1213 + num) + '.csv') as f:
            reader = csv.reader(f)
            data = []
            counter = 0
            for row in reader:
                if 0 < counter <= month_quantity - 1:
                    data.append(float(row[2]))
                counter += 1
        for i in [3, 2, 1, 0]:
            prediction = data[len(data) - i - 2]
            True_values.append(data[len(data) - i - 1])
            Predicted_values_1.append(prediction)
            if float(data[len(data) - i - 1]) == 0 and prediction == 0:
                acc = 1
            else:
                acc = min(data[len(data) - i - 1], prediction) / max(data[len(data) - i - 1], prediction)
            Acc_list.append(acc)
            data[len(data) - i - 1] = prediction
Average_Acc = sum(Acc_list)/len(Acc_list)
print(Average_Acc)
Predicted_values_2 = []
Acc_list = []
p = 1
for num in range(1146):
    with open('Time_Series/item' + str(1213 + num) + '.csv') as f:
        reader = csv.reader(f)
        month_quantity = 0
        for row in reader:
            month_quantity += 1
    if month_quantity > 4:
        with open('Time_Series/item' + str(1213 + num) + '.csv') as f:
            reader = csv.reader(f)
            data = []
            counter = 0
            for row in reader:
                if 0 < counter <= month_quantity - 1:
                    data.append(float(row[2]))
                counter += 1
        for i in [3, 2, 1, 0]:
            try:
                model = AutoReg(data[0:len(data) - i - 1], lags=p)
                model_fit = model.fit()
                prediction = model_fit.predict(len(data[0:len(data) - i - 1]), len(data[0:len(data) - i - 1]))
                if prediction[0] > 1:
                    prediction[0] = float(1)
                if prediction[0] < 0:
                    prediction[0] = float(0)
                Predicted_values_2.append(prediction[0])
                if float(data[len(data) - i - 1]) == 0 and prediction == 0:
                    acc = 1
                else:
                    acc = min(data[len(data) - i - 1], prediction) / max(data[len(data) - i - 1], prediction)
                Acc_list.append(acc)
                data[len(data) - i - 1] = prediction[0]
            except ZeroDivisionError:
                Acc_list.append(0)
                continue
            except ValueError:
                Acc_list.append(0)
                continue
Average_Acc = sum(Acc_list)/len(Acc_list)
print(Average_Acc)
Predicted_values_3 = []
Acc_list = []
q = 1
for num in range(1146):
    with open('Time_Series/item' + str(1213 + num) + '.csv') as f:
        reader = csv.reader(f)
        month_quantity = 0
        for row in reader:
            month_quantity += 1
    if month_quantity > 4:
        with open('Time_Series/item' + str(1213 + num) + '.csv') as f:
            reader = csv.reader(f)
            data = []
            counter = 0
            for row in reader:
                if 0 < counter <= month_quantity - 1:
                    data.append(float(row[2]))
                counter += 1
        for i in [3, 2, 1, 0]:
            try:
                model = ARMA(data[0:len(data) - i - 1], order=(0, q))
                model_fit = model.fit(disp=False)
                prediction = model_fit.predict(len(data[0:len(data) - i - 1]), len(data[0:len(data) - i - 1]))
                if prediction[0] > 1:
                    prediction[0] = float(1)
                if prediction[0] < 0:
                    prediction[0] = float(0)
                Predicted_values_3.append(prediction[0])
                if float(data[len(data) - i - 1]) == 0 and prediction == 0:
                    acc = 1
                else:
                    acc = min(data[len(data) - i - 1], prediction) / max(data[len(data) - i - 1], prediction)
                Acc_list.append(acc)
                data[len(data) - i - 1] = prediction[0]
            except ValueError:
                Acc_list.append(0)
                continue
            except numpy.linalg.LinAlgError:
                Acc_list.append(0)
                continue
            except ZeroDivisionError:
                Acc_list.append(0)
                continue
Average_Acc = sum(Acc_list)/len(Acc_list)
print(Average_Acc)
plt.plot(True_values[0:100], color='black', label = 'Original data')
plt.plot(Predicted_values_1[0:100], color='blue', label = 'Naive')
plt.plot(Predicted_values_2[0:100], color='green', label = 'AR')
plt.plot(Predicted_values_3[0:100], color='yellow', label = 'MA')
plt.legend(loc='best')
plt.title('Actual and predicted')
plt.show()