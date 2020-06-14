---
title:  "Using LSTMs for computer vision tasks"
header:
  teaser: "/assets/images/figures/lstm.png"
categories: 
  - deep learning
tags:
  - lstm
  - computer vision
classes: wide  
---

LSTM has always been widely successful in Natural Language Processing tasks. It is particularly designed for processing sequence and language is a sequence of words.

Let's say we have a dataset of `imdb reviews` and we want to classify them into positive reviews and negative ones. 

First, we encode all the words into their corresponding word embeddings. Word embeddings are unique vectors or 1d arrays that represent each word uniquely. Neural networks only deal with numerical values; not strings or characters. That's why representing each word with an embedding is necessary. Then these embeddings are fed into an LSTM cell one by one. The hidden state of the previous cell state is fed back again in the cell by a feedback loop. The output of the drawn from the last cell state.

<img src="{{site.url}}{{site.baseurl}}/assets/images/figures/lstm.png" alt="a figure of LSTM">

Same process can be used to process videos as videos are basically sequence of images or frames. So in tasks like `Action Recognition` or `Video Captioning` LSTM architectures can be utilized. This type of architecture are called `CNN-LSTM`s.

Vanilla LSTMs work with 1-dimensional embeddings or features. A pretrained CNN can work as an excellent feature extractor. Each frame of the video is passed into CNN and the values on Dense Layer before the final Classification Layer can be taken as 1d features. They are then passed inside LSTMs to learn sequence representations. 

Here is a diagram of a typical CNN.

<img src="{{site.url}}{{site.baseurl}}/assets/images/figures/cnn.png" alt="a figure of CNN">

Here is a diagram of basic CNN-LSTM architecture.

<img src="{{site.url}}{{site.baseurl}}/assets/images/figures/cnnlstm.png" alt="a figure of CNN-LSTM">

However, Vanilla 1d LSTM doesn't take into account spatial information. But,exploiting spatial features learned by CNNs can provide better spatio-temporal feature learning.

That is where `convLSTM2D` comes in. They take in 2d features instead of 1d and also output 2d features. In vanilla LSTM the inner gates or connections are connected by dense weights. But, in `convLSTM2D`, they are replaced with convolution operations. Because, connecting 2d layers with dense weights would cost huge number of parameters.This is same reason why in CNNs, the 2d feature layers are connected by convolution operations. 

<img src="{{site.url}}{{site.baseurl}}/assets/images/figures/cnnconvlstm.png" alt="a figure of CNN-convLSTM">

Here are the codes to build a basic `CNN-convLSTM` model in `Keras`. Each video input has 32 frames. Each has a shape of (224,224,3). So, input shape is (32,224,224,3). We can choose whether to train the pretrained CNN further using `trainable` variable.

```python

def CNN_ConvLSTMmodel():
  x = Input(shape=(32,224,224,3))
  cnn = MobileNet(include_top = false, input_shape = (224,224,3), weights = 'imagenet' )
  for layer in cnn.layers:
    layer.trainable = true
  x = TimeDistributed( cnn )(x)
  x = ConvLSTM2D( filters = 256, kernel_size=(3,3), return_sequence = False)(x)
  x = Flatten()(x)
  x = Dense(256, activation='relu')(x)
  predictions = Dense(number_of_classes, activation='softmax')(x)
  model = Model(inputs=[x], outputs=[predictions])
  return model

```

I hope from this article, we get a basic idea of how `CNN-LSTM` architectures work. To get more deeper understanding do visit the following links and try to play with them yourself.


1. [Understading LSTM - Colah's blog][colah-blog]
2. [CNN long short term memory][machine-learning-mastery]
3. [Convolutional LSTM paper][paper].

[colah-blog]: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
[machine-learning-mastery]:  https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
[paper]: https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf