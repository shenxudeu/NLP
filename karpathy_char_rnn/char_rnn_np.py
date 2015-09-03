"""
Derived from the Minimal character-level Vanilla RNN model written by Andrej Kapathy
Reference: https://gist.github.com/karpathy/d4dee566867f8291f086
"""

import numpy as np
from random import uniform
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
    Notes:
        - one loss value for the whole sequence, which is the sum of loss from each time-step
        - gradient calculation is like unfolding the 1 hidden layer network to multiple layers
        - we end up back-propogating a seq_length layers deep network
        - a trick to preventing exploding gradients to clip the gradient
    """
    xs, hs, ys, ps = {}, {}, {}, {} # input, hidden, output, out_prob states for each time t
    hs[-1] = np.copy(hprev)
    loss = 0
    
    # forward pass
    for t in xrange(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) 
        xs[t][inputs[t]] = 1. # convert input to one-hot
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t],0])
    
    # backward pass
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(xrange(len(inputs))):
        # backprop into y
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        # backprop into Why, hs, and by
        dWhy += np.dot(dy, hs[t].T)
        dby  += dy
        dh   = np.dot(Why.T, dy) + dhnext
        # backprop through tanh activition
        dhraw = (1 - hs[t] * hs[t]) * dh
        # backprop into Wxh, Whh, hs, and bh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    # clip gradient preventing exploding
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]


def sample(h, seed_ix, n):
    """
    generate a sequence of ints from a trained model
    Inputs:
        - h: initial hidden state
        - seed_ix: initial word
        - n: sequence length want to generate
    """
    x = np.zeros((vocab_size,1))
    x[seed_ix] = 1
    ixes = []
    for t in xrange(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size,1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


def gradCheck(inputs, targets, hprev):
    num_checks, delta = 10, 1e-5
    _, dWxh, dWhh, dWhy, dbh, dby, _ = lossFunc(inputs, targets, hprev)
    for param, dparam, name in zip([Wxh,   Whh,  Why,  bh,  by],
                                   [dWxh,  dWhh, dWhy, dbh, dby],
                                   ['Wxh', 'Whh', 'Why', 'bh', 'by']):
        s0 = dparam.shape
        s1 = dparam.shape
        assert s0 == s1, "Error dims don't match: %s and %s"%(`s0`, `s1`)
        print name
        for i in xrange(num_checks):
            ri = int(uniform(0, param.size))
            # evaluate loss at [x+delta] and [x-delta] 
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            cg0, _, _, _, _, _, _ = lossFunc(inputs, targets, hprev)
            param.flat[ri] = old_val - delta
            cg1, _, _, _, _, _, _ = lossFunc(inputs, targets, hprev)
            # reset back
            param.flat[ri] = old_val
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / (2*delta)
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_analytic + grad_numerical)
            print "%f, %f => %e"%(grad_numerical, grad_analytic, rel_error)
            # rel_error should be on order of 1e-7 or less


n, p  = 0, 0
# memory prameters, used for adagrad optimizer
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)

# Loss at iteration 0
smooth_loss = -np.log(1./vocab_size) * seq_length

while True:
    # prepare inputs, we are sweeping from left to right in steps seq_length long.
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size,1))
        p = 0 # reset pointer p to the begining
    inputs  = [char_to_ix[ch] for ch in data[p: p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1: p + seq_length+1]]

    # Graident Check
    #gradCheck(inputs, targets, hprev)
    #pdb.set_trace()
    
    # forward seq_length characters through the network and calculate gradients
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFunc(inputs, targets, hprev)
    smooth_loss = smooth_loss * .999 + loss * .001

    if n % 500 == 0:
        print 'Iteration %d, Loss: %.4f'%(n, smooth_loss)
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print '------\n %s \n ------'%(txt, )

    # SGD with Adagrad
    for param, dparam, mem in zip([Wxh,  Whh,  Why,  bh,  by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
    
    p += seq_length
    n += 1





