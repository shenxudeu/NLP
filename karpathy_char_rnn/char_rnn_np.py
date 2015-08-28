"""
Derived from the Minimal character-level Vanilla RNN model written by Andrej Kapathy
Reference: https://gist.github.com/karpathy/d4dee566867f8291f086
"""

import numpy as np
import pdb

# read data
data = open('paul_graham.txt','r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'The dataset contains %d characters, %d unique.'%(data_size, vocab_size)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyper-parameters
hidden_size = 100 
seq_length = 25
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # weight of input to hidden, (k x |v|)
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # weight of hidden to hidden, (k x k)
Why = np.random.randn(vocab_size, hidden_size)*0.01 # weight of hidden to output, (|v| x k)
bh  = np.zeros((hidden_size, 1)) # hidden bias
by  = np.zeros((vocab_size, 1)) # output bias

def lossFunc(inputs, targets, hprev):
    """
    The RNN forward and backward, this is for one mini-batch
    Parameters:
        - inputs: list of int, presenting a sequence of chars used as training input
        - targets: list of int, presenting a sequence of chars used as target
        - hprev: np array, (k x 1), k is the number of hidden node
    """
    xs, hs, ys, ps = {}, {}, {}, {} # input, hidden, output, out_prob states for each time t
    hs[-1] = np.copy(hprev)
    loss = 0
    
    # forward pass
    for t in xrange(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) 
        xs[t][inputs[t]] = 1. # convert input to one-hot

    pdb.set_trace()
    pass
    



