"""
Utility functions to help train models.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils


def train_bptt(decoder, contexts, targets, visible_seed, hidden_seed,
               optimizer, batch_size, bptt,
               cost_fn=nn.PairwiseDistance(), clip_grad_norm=1.):
    """
    Train a nn.Module object using given optimizer and training data sequences,
    using backpropagation through time (bptt), <bptt> sequence steps per
    optimization step.

    Unlike typical implementations, sequences in each batch are not required
    to have the same length, and no <blank> padding is done to enforce that.
    Shorter sequences are swapped out of the batch dynamically, replaced with
    next sequence in the data. This makes training more efficient.

    Trains one epoch, returns computed loss at each step, as a list.
    
    decoder:
    nn.Module object that implements a
      .forward(visible, context, hidden)
    method which takes in current visible features, a context vector, and
    current hidden state, and returns the next visible output features.
    i.e. Like a standard RNN which takes in an additional context vector at
    each step.

    contexts, targets:
    lists of sequence data; containing context vectors and target output values.
    contexts[i] must have same length as targets[i].
    i.e. input sequences must have same length as output sequences.

    visible_seed, hidden_seed:
    initial feature vector to jumpstart the sequence, and initial hidden state
    for start of the sequence. Of shape [<features>] and
    [<num_layers>, <n_hidden>] respectively.

    optimizer:
    a torch.optim object, e.g. initialized by torch.optim.Adam(...).

    batch_size, bptt:
    Number of sequences to train concurrently, and number of steps to proceed
    in sequence before calling an optimization step.
    """

    # indices holds sequence index in each slot of the batch.
    indices = np.arange(batch_size)

    # positions holds, for each sequence in the batch, the position where
    # we are currently at.
    positions = [0 for i in range(batch_size)]
    
    visible = torch.stack([visible_seed for i in range(batch_size)]).unsqueeze(0)
    hiddens = torch.stack([hidden_seed for i in range(batch_size)], 1)

    losses = []
    marker = batch_size
    while marker < len(targets):
        optimizer.zero_grad()
        # The following two lists hold output of the decoder, and
        # the values they are to be costed against.
        predicted, actual = [], []
        for counter in range(bptt):
            inputs = torch.stack([contexts[i][p]
                                  for i, p in zip(indices, positions)])
            outputs, hiddens = decoder(visible, inputs, hiddens)
            predicted.append(outputs[0])
            visible = outputs.clone() # can implement teacher forcing here.
            actual.append(torch.stack([targets[i][p]
                                       for i, p in zip(indices, positions)]))

            for b, index, position in zip(list(range(batch_size)),
                                          indices, positions):
                if len(targets[index]) > position + 1:
                    positions[b] += 1                           
                else: #load next sequence
                    marker += 1
                    # we wrap around to start of dataset, if some long
                    # seqence near end of dataset isn't done yet.
                    indices[b] = marker % len(targets)
                    positions[b] = 0
                    visible[0, b] = visible_seed
                    hiddens[:, b] = hidden_seed

        loss = torch.mean(cost_fn(torch.cat(predicted), torch.cat(actual)))
        loss.backward()
        torch.nn.utils.clip_grad_norm(decoder.parameters(), clip_grad_norm)
        optimizer.step()
        losses.append(loss.data[0])
        
        # The following frees up memory, discarding computation graph
        # in between optimization steps.
        visible = visible.detach()
        hiddens = hiddens.detach()
        
    return losses
