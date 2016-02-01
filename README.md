# SNLI task with LSTM Memory Network encoder-dencoder and neural attention
This is an implementation for the deep attention fusion LSTM memory network presented in the paper "[Long Short-Term Memory Networks for Machine Reading](http://arxiv.org/abs/1601.06733)" 
Please note that it is a research code.

## Setup and Usage
This code requires [Torch7](http://torch.ch/) and [nngraph](http://github.com/torch/nngraph). It is tested on GPU only.

To run the code execute: th LSTMN.lua -gpuid 0


## Lookup Tables
There are different ways to use pre-trained word vectors, which can be explored.

* [LookupTableEmbedding_fixed]: Initialize with pre-trained word vectors. Words not in the coverage are randomly initialized. Word vectors are not updated in the training.

* [LookupTableEmbedding_train]: Same as above but all word vectors are updated during training.

* [LookupTableEmbedding_update]: Same as above but the gradient of words with pre-trained vectors will be scaled by a factor of 0.1 .
