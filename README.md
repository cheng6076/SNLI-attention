# SNLI task with LSTM Memory Network encoder-dencoder and neural attention

This is an implementation for the deep attention fusion LSTM memory network presented in the paper "[Long Short-Term Memory Networks for Machine Reading](http://arxiv.org/abs/1601.06733)". 

Please note that it is a research code and most hyper-parameters remain untuned. With the current setting, it is expected to get the best number around 0.86 with Glove 840b vectors. There are many reasons why you may get different numbers, one of which is that the vectors for OOV words are randomly initialized. To reduce the stochasticity, pre-compute the initial vector of each OOV word as the average of its neighboring words in the dataset (though computing the vectors for OOV words in dev/test set after training is even better). The OOV vecs can be obtained with oov_vec.py, and then one can cat the OOV vector and the original vector into a single file.

To do: add the mask for the memory; move batch_size out of the attention module.

## Setup and Usage
This code requires [Torch7](http://torch.ch/) and [nngraph](http://github.com/torch/nngraph). It is tested on GPU only.

To run the code execute: th LSTMN.lua -gpuid 0


## Lookup Tables
There are different ways to use pre-trained word vectors, which has not been fully explored yet.

* [LookupTableEmbedding_fixed]: Initialize with pre-trained word vectors. Words not in the coverage are randomly initialized. Word vectors are not updated in the training.

* [LookupTableEmbedding_train]: Same as above but all word vectors are updated during training.

* [LookupTableEmbedding_update]: Same as above but only words without pre-trained vectors get updated during training, as used in Rocktäschel et al., 2015. An alternative is to scale down the gradient of non-OOV words by a factor of k. 
