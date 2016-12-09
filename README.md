# SNLI task with LSTM Memory Network encoder-dencoder and neural attention

This is an implementation for the deep attention fusion LSTM memory network presented in the paper "[Long Short-Term Memory Networks for Machine Reading](EMNLP,2016)". 

## Setup and Usage
This code requires [Torch7](http://torch.ch/) and [nngraph](http://github.com/torch/nngraph). 
It is updated to use torch version around May 2016. Minimum preprocessing is needed to obtain a good accuracy, including lower-casing and tokenization.  

## Citation
```
@article{cheng2016,
  author = {Cheng, Jianpeng and Dong, Li and Lapata, Mirella,
  title = {Long Short-Term Memory Networks for Machine Reading},
  journal = {EMNLP},
  year = {2016},
  pages = {551--562}
}

```
