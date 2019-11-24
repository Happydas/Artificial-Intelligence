
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(inputs):
    """
    Calculate the sigmoid for the give inputs (array)
    :param inputs:
    :return:
    """
    #The code 1 / float(1 + np.exp(- x)) is the fucuntion is used for calcualting the sigmoid scores.
    sigmoid_scores = [1 / float(1 + np.exp(- x)) for x in inputs]
    return sigmoid_scores


def line_graph(x, y, x_title, y_title):
    """
    Draw line graph with x and y values
    :param x:
    :param y:
    :param x_title:
    :param y_title:
    :return:
    """
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

#Creating a graph_x list which contains the numbers in the range of 0 to 21.
graph_x = range(0, 21)
#  graph_y list storing the calculated sigmoid scores for the given graph_x inputs.
graph_y = sigmoid(graph_x)

print("Graph X readings: {}".format(graph_x))
print("Graph Y readings: {}".format(graph_y))

#Calling the line_graph function, which takes the x, y, and titles of the graph to create the line graph.
line_graph(graph_x, graph_y, "Inputs", "Sigmoid Scores")

