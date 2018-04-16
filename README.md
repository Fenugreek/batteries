# batteries
Tools to assist modeling and training with PyTorch.

Currently, one useful training algorithm is offered: training RNNs with backpropagation-through-time without requiring training sequences to be of the same length, dynamically swapping out sequences in the batch as they get exhausted.

See doc string for `train_bptt()` in `train.py` for more details. Also, my blog post, [Better Backprop Through Time Using Pytorch](http://subburam.org/2018/04/better-backprop-through-time-using-pytorch/) for some discussion.
