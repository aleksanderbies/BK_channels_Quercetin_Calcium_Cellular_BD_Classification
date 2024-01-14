import tensorflow as tf
import torch
import torch.nn as nn

# Definicja modelu RNN
class SimpleRNN(tf.keras.Model):
    def __init__(self, units):
        super(SimpleRNN, self).__init__()
        self.units = units
        self.rnn_cell = tf.keras.layers.SimpleRNNCell(units)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, states=None, return_state=False):
        x, states = self.rnn_cell(inputs, states)
        output = self.dense(x)
        if return_state:
            return output, states
        return output

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, input, hidden_state):
        hidden_state = self.gru_cell(input, hidden_state)
        output = self.output_layer(hidden_state)
        return output, hidden_state