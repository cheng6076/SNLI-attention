# SNLI task with LSTM encoder-dencoder and neural attention
This implements some baselines for the Standford Natural Language Inference task with the seq2seq model.

* [LSTM-attention.lua] A simple word-word attention model. The model computes the decoders hidden states and the corresponding attentional encoder hidden states, performs average pooling to each, concatenate the result and feed them into a NN classifier. I used pre-trained word2vec to initialize the word vectors. Only those without pre-trained vectors are updated during training.

* [LSTM-nonattention.lua] A non-attention baseline is implemented where all vectors are randomly initialized and updated during training.

I used online training to process one pair at a time, to make the computation of sequences with different lengths more accurate. But one can also use mini-batch training.

## Lookup Tables
There are different ways to use pre-trained word vectors, which can be explored.

* [LookupTableEmbedding_fixed]: Initialize with pre-trained word vectors. Words not in the coverage are randomly initialized. Word vectors are not updated in the training.

* [LookupTableEmbedding_train]: Same as above but all word vectors are updated during training.

* [LookupTableEmbedding_update]: Same as above but only words without pre-trained vectors get updated during training, as used in Rocktäschel et al., 2015.
