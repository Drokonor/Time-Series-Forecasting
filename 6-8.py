from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import csv
from arch import arch_model
import matplotlib.pylab as plt
True_values = []
Predicted_values_1 = []
Acc_list = []
Smooth = 0.26
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
                model = SimpleExpSmoothing(data[0:len(data) - i - 1])
                model_fit = model.fit(smoothing_level=Smooth)
                prediction = model_fit.predict(len(data[0:len(data) - i - 1]), len(data[0:len(data) - i - 1]))
                if prediction[0] > 1:
                    prediction[0] = float(1)
                if prediction[0] < 0:
                    prediction[0] = float(0)
                True_values.append(data[len(data) - i - 1])
                Predicted_values_1.append(prediction[0])
                if float(data[len(data) - i - 1]) == 0 and prediction == 0:
                    acc = 1
                else:
                    acc = min(data[len(data) - i - 1], prediction) / max(data[len(data) - i - 1], prediction)
                Acc_list.append(acc)
                data[len(data) - i - 1] = prediction[0]
            except IndexError:
                Acc_list.append(0)
                continue
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
                model = arch_model(data[0:len(data) - i - 1], mean='Zero', vol='ARCH', p=p, rescale=True)
                model_fit = model.fit()
                prediction_0 = model_fit.forecast(horizon=1)
                prediction = prediction_0.variance.values[len(prediction_0.variance.values) - 1][0]
                if prediction > 1:
                    prediction = float(1)
                if prediction < 0:
                    prediction = float(0)
                Predicted_values_2.append(prediction)
                if float(data[len(data) - i - 1]) == 0 and prediction == 0:
                    acc = 1
                else:
                    acc = min(data[len(data) - i - 1], prediction) / max(data[len(data) - i - 1], prediction)
                Acc_list.append(acc)
                data[len(data) - i - 1] = prediction
            except ValueError:
                Acc_list.append(0)
                continue
Average_Acc = sum(Acc_list)/len(Acc_list)
print(Average_Acc)
Predicted_values_3 = []
Acc_list = []
p = 4
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
                model = arch_model(data[0:len(data) - i - 1], mean='Zero', vol='GARCH', p=p, q=q, rescale=False)
                model_fit = model.fit()
                prediction_0 = model_fit.forecast(horizon=1)
                prediction = prediction_0.variance.values[len(prediction_0.variance.values) - 1][0]
                if prediction > 1:
                    prediction = float(1)
                if prediction < 0:
                    prediction = float(0)
                True_values.append(data[len(data) - i - 1])
                Predicted_values_3.append(prediction)
                if float(data[len(data) - i - 1]) == 0 and prediction == 0:
                    acc = 1
                else:
                    acc = min(data[len(data) - i - 1], prediction) / max(data[len(data) - i - 1], prediction)
                Acc_list.append(acc)
                data[len(data) - i - 1] = prediction
            except ValueError:
                Acc_list.append(0)
                continue
            except ZeroDivisionError:
                Acc_list.append(0)
                continue
Average_Acc = sum(Acc_list)/len(Acc_list)
print(Average_Acc)
plt.plot(True_values[0:100], color='black', label = 'Original data')
plt.plot(Predicted_values_1[0:100], color='blue', label = 'SES')
plt.plot(Predicted_values_2[0:100], color='green', label = 'ARCH')
plt.plot(Predicted_values_3[0:100], color='yellow', label = 'GARCH')
plt.legend(loc='best')
plt.title('Actual and predicted')
plt.show()