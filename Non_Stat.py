from pandas import Series
from statsmodels.tsa.stattools import kpss
import csv
import matplotlib.pylab as plt
Non_Stat = []
for num in range(1146):
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
        series = Series(data)
        result = kpss(series, regression='c')
        Non_Stat.append(result[1])
graph = []
for i in range(len(Non_Stat)):
    graph.append(0.02)
plt.plot(graph, color='black', label = 'Уровень значимости')
plt.plot(Non_Stat, color='blue', label = 'p_value')
plt.legend(loc='best')
plt.title('KPSS_test')
plt.show()