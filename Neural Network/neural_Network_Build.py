# Package imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

# load the data from the file
data = load_data("../Data_ML/marks.txt", None)

# X = feature values, all the columns except the last column
input = data.iloc[:, :-1]
print('input:', input.shape)

# y = target values, last column of the data frame
target = data.iloc[:, -1]
print('target:', target.shape)
#the newaxis expression is used to increase the dimension of the existing array by one more dimension, when used once.
output = target[:, np.newaxis]  # size of output layer
print('output:', output.shape)

# filter out the applicants that got admitted
admitted = data.loc[target == 1]

# filter out the applicants that din't get admission
not_admitted = data.loc[target == 0]


def neural_Layer(X, y):
    """
        Arguments:
        X -- input dataset of shape (number of examples, input size)
        y -- labels of shape (number of examples, output size)

        Returns:
        X -- the size of the input layer
        hidden -- the size of the hidden layer
        y -- the size of the output layer
        """
    #print("X.Shape:", X.shape)
    X = X.shape[1]
    #print('X:', X)
    hidden = 4
    #print("y.Shape:", y.shape)
    y = y.shape[1] #size of output layer
    #print("y:", y)
    return X, hidden, y

X, hidden, y = neural_Layer(input, output)

def initialize_parameters(X, hidden, y):
    """
    Argument:
    X -- size of the input layer
    hidden -- size of the hidden layer
    y -- size of the output layer

    Returns:
    params -- python dictionary containing the parameters:
                    W1 -- weight matrix of shape (X, hidden)
                    b1 -- bias vector of shape (1, hidden)
                    W2 -- weight matrix of shape (hidden, y)
                    b2 -- bias vector of shape (1, y)
    """

    Weight1 = np.random.rand(X, hidden)
    print('Weight1.shape:', Weight1.shape)
    bias1 = np.zeros((1, hidden))
    print('bias1.shape:', bias1.shape)
    Weight2 = np.random.rand(hidden, y)
    print('Weight2.shape:', Weight2.shape)
    bias2 = np.zeros((1, y))
    print('bias2.shape:', bias2.shape)

    parameters = {"W1": Weight1,
                  "b1": bias1,
                  "W2": Weight2,
                  "b2": bias2}

    return parameters
parameters = initialize_parameters(X, hidden, y)

# forward_propagation
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (m, X)
    parameters -- python dictionary containing parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Implement Forward Propagation to calculate A2 (probabilities)
    #print("X.Shape:", X.shape)
    #print("W1.Shape:", W1.shape)
    #print("b.Shape:", b1.shape)
    #print("output size:", y.shape)
    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)
    Z2A = np.dot(A1, W2)
    Z2 = np.dot(A1, W2) + b2
    #print("Z2", Z2)
    A2 = sigmoid(Z2)
    #print("A2", A2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache
A2, cache = forward_propagation(input, parameters)
print('A2.shape', A2.shape)


#compute_cost

def compute_cost(A2, Y, parameters):
    """

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (number of examples,1)
    Y -- "true" labels vector of shape (number of examples, 1)
    parameters -- python dictionary containing the parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost
    """

    m =   input.shape[0]  # number of example
   # print('m:', m)
    #print('A2.shape', A2.shape)
    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    # np.squeeze Remove single-dimensional entries from the shape of an array.
    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    #print("cost:", cost)

    return cost
#cost = compute_cost(A2, output, parameters)

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (number of examples, 2)
    Y -- "true" labels vector of shape (number of examples, 1)

    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m =  input.shape[0]
    #print('m: number', m)

    # Retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters['W1']
    W2 = parameters['W2']

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    #print('A1.shape', A1.shape)
    A2 = cache['A2']

    # Backward propagation: calculate dW1, db1, dW2, db2.

    dZ2 = A2 - Y
    #print('dZ2.shape', dZ2.shape)
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    #print('dW2.shape', dW2.shape)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    #print('db2.shape', db2.shape)

    dZ1 = np.multiply(np.dot(dZ2, W2.T), 1 - np.power(A1, 2))
    #dZ1 = np.dot(dZ2, W2.T)
    #print('A1.shape', A1.shape)
    #print('dZ1.shape', dZ1.shape)
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    #print('dW1.shape', dW1.shape)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    #print('db1.shape', db1.shape)

    gradient = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return gradient

#grads = backward_propagation(parameters, cache, input, output)

def update_parameters(parameters, grads, learning_rate=1.25):
    """
    Updates parameters using the gradient descent update rule

    Arguments:
    parameters -- python dictionary containing the parameters
    grads -- python dictionary containing the gradients

    Returns:
    parameters -- python dictionary containing the updated parameters
    """

    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # Update rule for each parameter

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters
#paremeters = update_parameters(parameters, grads, learning_rate=1.25)

def nn_model(input, output, hidden, num_iterations=3400, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (number of examples, 2)
    y -- labels of shape (number of examples, 1)
    hidden -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(3)
    X = neural_Layer(input, output)[0]
    y = neural_Layer(input, output)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "X, hidden, y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(X, hidden, y)

    # Loop (gradient descent)
    epoch = []
    error = []

    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(input, parameters)
        #print('A2:', A2)

        # Cost function. Inputs: "A2, output, parameters". Outputs: "cost".
        cost = compute_cost(A2, output, parameters)

        # Backpropagation. Inputs: "parameters, cache, input, output". Outputs: "grads".
        grads = backward_propagation(parameters, cache, input, output)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)

        error.append(cost)
        epoch.append(i)

        # Print the cost every 200 iterations
        if print_cost and i % 200 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return parameters, error, epoch

def predict(parameters, input):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing the parameters
    input -- input data of size (m, X)

    Returns
    predictions -- vector of predictions of the model
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(input, parameters)
    predictions = A2 > 0.5
    return predictions

example = np.array([[82.30705337399482, 76.48196330235604]])
prediction = (predict(parameters,example))
example_2 = np.array([[0.08327747668339, 16.3163717815305]])
prediction_2 = (predict(parameters,example_2))
print('Correct Prediction: ', prediction)
print('Correct Prediction_2: ', prediction_2)
print(predict(parameters,example_2), ' - Correct: ', example_2[0][0])

parameters, error, epoch = nn_model(input, output, hidden = 4, num_iterations = 3400, print_cost=True)

# plot the error over the entire training duration
plt.figure(figsize=(15,8))
plt.plot(epoch, error)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()




def plot_decision_boundary(model, input, y):
    # Set min and max values and give it some padding
    x_min, x_max = input[:, 0].min() - 1, input[:, 0].max() + 1
    y_min, y_max = input[:, 1].min() - 1, input[:, 1].max() + 1
    #x_min, x_max = input[0, :].min() - 1, input[0, :].max() + 1
    #y_min, y_max = input[1, :].min() - 1, input[1, :].max() + 1
    h = 1
    # Generate a grid of points with distance h between them
    #xx, yy = np.meshgrid(np.arange(30, 80, h), np.arange(30, 80, h))
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #Z = Z.np.linspace(start=1, stop=2500, num=2500)
    # Plot the contour and training exampl
    #cmap : Colormap, optional, default: None
    plt.contour(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    #plt.scatter(input[0, :], input[1, :], c=y, s= 10, cmap=plt.cm.Spectral)
    plt.scatter(input[:, 0], input[:, 1], c=y, s=10, cmap=plt.cm.Spectral)






