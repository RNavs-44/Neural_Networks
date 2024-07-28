Neural Networks Zero to Hero:
micrograd:
-> A tiny Autograd engine. Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG (Directed Acrylic Graph) and a small neural networks library on top of it with a PyTorch-like API. 
-> Both are tiny, with about 100 and 50 lines of code respectively. 
-> The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification.

makemore:
-> makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. 
-> Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT). 
-> For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble.

key papers:
-> Bigram (one character predicts the next one with a lookup table of counts)
-> MLP, following Bengio et al. 2003
-> CNN, following DeepMind WaveNet 2016 (in progress...)
-> RNN, following Mikolov et al. 2010
-> LSTM, following Graves et al. 2014
-> GRU, following Kyunghyun Cho et al. 2014
-> Transformer, following Vaswani et al. 2017
