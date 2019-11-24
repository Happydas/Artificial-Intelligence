import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_data():
    # To read csv file
    btc = pd.read_csv('../Data/btc.csv')
    btc.head()
    # print(btc)

    # selecting only the column that we are going to use in the prediction process
    data_to_use = btc['Close'].values
    print(data_to_use)
    print("length", len(data_to_use))

    # data preprocessing(scaling)
    #Standardize the data to bring it to the same scale
    scaler = StandardScaler()
    # Reshaping a tensor means rearranging its rows and columns to match a target shape.
    # Naturally, the reshaped tensor has the same total number of coefficients as the initial tensor.
    # Reshape your data either X.reshape(-1, 1) if your data has a single feature/column and # X.reshape(1, -1) if it contains a single sample.
    scaled_data = scaler.fit_transform(data_to_use.reshape(-1, 1))
    # print(scaled_data)
    return scaled_data


# This function is used to create Features and Labels datasets. By windowing the data.
# [Example: if window_size = 1 we are going to use only the previous day to predict todays stock prices]
# Outputs: X - features splitted into windows of datapoints (if window_size = 1, X = [len(data)-1, 1])
# y - 'labels', actually this is the next number in the sequence, this number we are trying to predict
def window_data(data, window_size):
    """
    :param data: dataset used in the project
    :param window_size: how many data points we are going to use to predict the next datapoint in the sequence
    :return: Generates the input and target columns
    """
    X = []
    y = []

    i = 0
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

        i += 1
    assert len(X) == len(y)
    """
    Python has built-in assert statement to use assertion condition in the program. 
    assert statement has a condition or expression which is supposed to be always true. If the condition is false 
    assert halts the program and gives an AssertionError .
    """
    return X, y

scaled_data = load_data()

plt.figure(figsize=(10,7))
plt.title("Bitcoin prices from 2014 to 2018")
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.plot(scaled_data, label = 'Price')
plt.legend()
plt.show()
