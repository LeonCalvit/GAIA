import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import random


def plot_lum_temp(show):
    with open('GAIA_DATA.csv', 'r') as csv_file:
        file_reader = csv.reader(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        x = np.empty([4001, 1], dtype=np.float)
        y = np.empty([4001, 1], dtype=np.float)
        f = 0
        my_list = []
        for row in file_reader:
            my_list.append([row[4], row[5]])
        random.shuffle(my_list)
        for row in my_list:
            x_temp = row[0]
            y_temp = row[1]
            if x_temp != '' and y_temp != '':
                x[f] = x_temp
                y[f] = y_temp
                f += 1
            if f > 4000:
                break
        plt.scatter(x, y, marker='.', alpha=0.2)
        plt.xscale('log')
        plt.xlabel('Temperature')
        plt.gca().invert_xaxis()
        plt.yscale('log')
        plt.ylabel('Luminosity')
        if show:
            plt.show()
        else:
            plt.savefig('diagram1.png')


def regression(attr1, attr2, model, take_log=False):
    with open('GAIA_DATA.csv', 'r') as csv_file:
        file_reader = csv.reader(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        X = []
        y = []
        for row in file_reader:
            temp_x = row[attr1]
            temp_y = row[attr2]
            if temp_x != '' and temp_y != '':
                X.append(temp_x)
                y.append(temp_y)
        X = np.array(X, dtype=np.float).reshape(-1, 1)
        y = np.array(y, dtype=np.float).reshape(-1, 1)
        if take_log:
            X = np.log10(X)
            y = np.log10(y)
        reg = model.fit(X, y)
        print(reg.score(X, y))


def cluster(plot):
    with open('GAIA_DATA_small.csv', 'r') as csv_file:
        file_reader = csv.reader(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        X = []
        y = []
        for row in file_reader:
            temp_x = row[1]
            temp_y = row[2]
            if temp_x != '' and temp_y != '':
                X.append(temp_x)
                y.append(temp_y)
        X = np.array(X, dtype=np.float).reshape(-1, 1)
        y = np.array(y, dtype=np.float).reshape(-1, 1)
        data = np.hstack((X, y))
        train, test = train_test_split(X, shuffle=False)
        for i in range(10, 50, 5):
            kmeans = KMeans(n_clusters=i).fit(train)
            print(kmeans.score(test))
        if plot:
            # Plotting the data.  No obvious real clusters
            plt.scatter(X, y, alpha=0.1, marker='.')
            plt.show()


if __name__ == "__main__":
    luminosity = 4
    temperature = 5
    radius = 7
    my_dict = {7: 'Radius', 5: 'Temperature', 4: 'Luminosity'}
    plot_lum_temp(True)
    print("Temp vs. Luminosity linear regression correlation:")
    regression(luminosity, temperature, LinearRegression(), take_log=True)
    print("Temperature vs. radius linear regression correlation:")
    regression(temperature, radius, LinearRegression())
    print("Luminosity vs. Radiuslinear regression correlation:")
    regression(luminosity, radius, LinearRegression())

    for x in [[temperature, radius], [luminosity, radius], [luminosity, temperature]]:
        for i in range(2, 10):
            print(f"{my_dict[x[0]]} vs. {my_dict[x[1]]} quadratic regression correlation with degree {i}:")
            model = make_pipeline(PolynomialFeatures(i), Ridge())
            log = (x[0] == luminosity) and (x[1] == temperature)
            regression(x[0], x[1], model, take_log=log)
    cluster(False)
