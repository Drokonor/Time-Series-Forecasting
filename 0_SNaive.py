# Наивная сезонная модель
# 2 -> 0.5439297412891577
# 3 -> 0.5562665270661191
# 4 -> 0.5463118785896962
import csv
import matplotlib.pylab as plt
Acc_list = []
True_values = []
Predicted_values = []
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
    if len(data) > 4:
        for i in [3, 2, 1, 0]:
            prediction = data[len(data) - i - 5]
            True_values.append(data[len(data) - i - 1])
            Predicted_values.append(prediction)
            if float(data[len(data) - i - 1]) != 0 and prediction != 0:
                acc = min(data[len(data) - i - 1], prediction)/max(data[len(data) - i - 1], prediction)
            else:
                acc = 1
            Acc_list.append(acc)
            data[len(data) - i - 1] = prediction
Average_Acc = sum(Acc_list)/len(Acc_list)
print(Average_Acc)
plt.plot(True_values, color='black', label = 'Original data')
plt.plot(Predicted_values, color='blue', label = 'Predicted data')
plt.legend(loc='best')
plt.title('Actual and predicted')
plt.show()