#importing the libraries
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


hidden_layer = 256
#weights and implementation of LSTM cell
#Weights for the input gate
#tf.truncated_normal() selects random numbers from a normal distribution whose mean is close to 0 and values are close to
weights_input_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
print('weight_input_gate.shape:', weights_input_gate.shape)
weights_input_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
print('weights_input_hidden.shape:', weights_input_hidden.shape)
#This operation set all elements to zero
bias_input = tf.Variable(tf.zeros([hidden_layer]))
print('bias_input.shape:', bias_input.shape)

#weights for the forgot gate
weights_forget_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_forget_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_forget = tf.Variable(tf.zeros([hidden_layer]))

#weights for the output gate
weights_output_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_output_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_output = tf.Variable(tf.zeros([hidden_layer]))

#weights for the memory cell
weights_memory_cell = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_memory_cell_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_memory_cell = tf.Variable(tf.zeros([hidden_layer]))


# function to compute the gate states
#LSTM_cell returns the cell state and hidden state as an output
def LSTM_cell(input, prev_hid_state, prev_cell_state):
    """

    :param input: current input
    :param prev_hid_state: previous hidden state
    :param prev_cell_state: previous cell state
    :return: returns the current cell state and current hidden state as an output
    """
    input_gate = tf.sigmoid(tf.matmul(input, weights_input_gate) + tf.matmul(prev_hid_state, weights_input_hidden) + bias_input)

    forget_gate = tf.sigmoid(tf.matmul(input, weights_forget_gate) + tf.matmul(prev_hid_state, weights_forget_hidden) + bias_forget)

    output_gate = tf.sigmoid(tf.matmul(input, weights_output_gate) + tf.matmul(prev_hid_state, weights_output_hidden) + bias_output)

    memory_cell = tf.tanh(tf.matmul(input, weights_memory_cell) + tf.matmul(prev_hid_state, weights_memory_cell_hidden) + bias_memory_cell)

    cell = prev_cell_state * forget_gate + input_gate * memory_cell

    hidden = output_gate * tf.tanh(cell)
    return cell, hidden

