import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import define_Data
import define_LSTM_Gate

scaled_data = define_Data.load_data()
#windowing the data with window_data function
X, y = define_Data.window_data(scaled_data, 7)


#we now split the data into training and test set
X_train  = np.array(X[:1018])
y_train = np.array(y[:1018])

X_test = np.array(X[1018:])
y_test = np.array(y[1018:])

#print("X size: {}".format(X.shape))
#print("y size: {}".format(y.shape))

print("X_train size: {}".format(X_train.shape))
#X_train size: (1018, 7, 1). shape implies that the sample_size, time_steps, and features function and the LSTM network
#require input exactly as follows. 1018 sets the number of data points(sample_size), 7 specifies the window size(time_steps)
# 1 specifies the dimention of our dataset(features)
print("y_train size: {}".format(y_train.shape))
print("X_test size: {}".format(X_test.shape))
print("y_test size: {}".format(y_test.shape))

batch_size = 7 #how many windows of data we are passing at once
window_size = 7  #how big window_size is (Or How many days do we consider to predict next point in the sequence)
hidden_layer = 256 #How many units do we use in LSTM cell
clip_margin = 4 #To prevent exploding gradient, we use clipper to clip gradients below -margin or above this margin
learning_rate = 0.001 #This is a an optimization method that aims to reduce the loss function.
#Learning Rate is a parameter of the Gradient Descent algorithm which helps us control
#the change of weights for our network to the loss of gradient.
epochs = 200 #one forward pass and one backward pass of all the training examples, This is the number of iterations (forward and back propagation) our model needs to make.

#Placeholders allows us to send different data within our network with the tf.placeholder() command.
inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1])
targets = tf.placeholder(tf.float32, [batch_size, 1])
print("input shape:", inputs.shape)
print("target shape:", targets.shape)
#Output layer weigts
weights_output = tf.Variable(tf.truncated_normal([hidden_layer, 1], stddev=0.05))
bias_output_layer = tf.Variable(tf.zeros([1]))

#perform forward propagation to predict the output.
#  A list is initialized to store the predicted output
outputs = []
#for each iteration output is computed and stored in the outputs list
for i in range(batch_size):  # Iterates through every window in the batch. The Batch Size refers to the number of training samples propagated through the network
    # for each batch creating batch_state as all zeros and output for that window which is all zeros at the beginning as well.
    #initialize hidden state and cell state. np.zeros() Return a new array of given shape and type, filled with zeros.
    cell_state = np.zeros([1, hidden_layer], dtype=np.float32)
    hidden_state = np.zeros([1, hidden_layer], dtype=np.float32)
    #print("hidden state:", hidden_state)

    # for each point in the window we are feeding that into LSTM to get next output
    #perform the forword propagation and compute the hidden and cell state of the LSTM cell for each time step
    for ii in range(window_size):
        #With reshape, a method in TensorFlow, we can change the dimensionality of tensors.
        # This means we flatten arrays (or rearrange values to be 2D).
        batch_state, batch_output = define_LSTM_Gate.LSTM_cell(tf.reshape(inputs[i][ii], (-1, 1)), hidden_state, cell_state)
        #print("inputs[i][ii]:", inputs[i][ii].shape)
    # last output is conisdered and used to get a prediction
    outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)

#Define backpropagation. A procedure to repeatedly adjust the weights so as to minimize the difference between actual output and desired output.
# After performing forward propagation and predicting the output, loss is computed with mean squared error loss function
#total loss is the sum of all loses across all of the time steps
losses = []

for i in range(len(outputs)):
    #Mean square error is measured as the average squared difference between predictions and actual observations.
    losses.append(tf.losses.mean_squared_error(tf.reshape(targets[i], (-1, 1)), outputs[i]))
# This function finds the mean
loss = tf.reduce_mean(losses)

#The gradient(derivative) of the slope tells us the direction we need to move towards to reach the minima
# Get all TensorFlow variables marked as "trainable" (i.e. all of them except learning rate)
#tvars = tf.trainable_variables(). get all variables that need to be optimized.# we define optimizer with gradient clipping.
gradients = tf.gradients(loss, tf.trainable_variables())
# Define the gradient clipping threshold
# to avoid the expoding gradient problem, gradient clipping is used
clipped, _ = tf.clip_by_global_norm(gradients, clip_margin)
#Create the Adam (Adaptive Moment Estimation) optimizer with our learning rate to minimize the loss, which is an extension to stochastic gradient descent algorithm.
#Gradient descent is an optimization algorithm for finding the minimum of a function. Learning rate is an optimization method that aims to reduce the loss function
optimizer = tf.train.AdamOptimizer(learning_rate)
# Create the training TensorFlow Operation through our optimizer. Apply gradients to variables.
trained_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
# Training the LSTM model.  initialization operation after having launched the graph.
#start the tensorflow session and initialize all the global variables. Variables must be initialized by running an
session = tf.Session()
session.run(tf.global_variables_initializer())
for i in range(epochs):
    traind_scores = []
    ii = 0
    epoch_loss = []
    # sample the number of data and train the network
    while (ii + batch_size) <= len(X_train):
        X_batch = X_train[ii:ii + batch_size]
        y_batch = y_train[ii:ii + batch_size]
        #feed_dict is used to pass the values for the placeholders
        o, c, _ = session.run([outputs, loss, trained_optimizer], feed_dict={inputs: X_batch, targets: y_batch})

        epoch_loss.append(c)
        traind_scores.append(o)
        ii += batch_size
    if (i % 30) == 0:
        #pass
        print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))


sup =[]
for i in range(len(traind_scores)):
    for j in range(len(traind_scores[i])):
        sup.append(traind_scores[i][j][0])
print("Sup:", sup)

tests = []
# to start making predictions on the test set
i = 0
while i + batch_size <= len(X_test):
    o = session.run([outputs], feed_dict={inputs: X_test[i:i + batch_size]})
    i += batch_size
    tests.append(o)
print("Test:", tests[0])

# The values of the test predictions are in a nested list, so we should flatten them
tests_new = []
for i in range(len(tests)):
    for j in range(len(tests[i][0])):
        tests_new.append(tests[i][0][j])
# now the predicted values are no longer in a nested list
print("test_New:", tests_new[0])

test_results = []
# took first 1019 points a s a training, so need to make predictions for the steps greater then 1019
for i in range(1264):
      if i >= 1019:
        test_results.append(tests_new[i-1019])
      else:
        test_results.append(None)
#we now plot predictions from the network
print("Test Results:", test_results)
plt.figure(figsize=(16, 7))
plt.title('Prices for bitcoin')
plt.xlabel('Days')
plt.ylabel('Scaled Price of Bitcoin')
plt.plot(scaled_data, label='Original data')
plt.plot(sup, label='Training data')
plt.plot(test_results, label='Testing data')
plt.legend()
plt.show()