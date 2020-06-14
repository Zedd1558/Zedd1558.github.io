---
title:  "Using LSTMs for computer vision tasks"
header:
  teaser: "/assets/images/500x300.png"
categories: 
  - deep learning
tags:
  - lstm
  - computer vision
classes: wide  
---

LSTM has always been widely successful in Natural Language Processing. It is particularly designed for processing sequence and language is a sequence of words.

Let's say we have a dataset of `imdb reviews` and we want to classify them into positive reviews and negative ones. First we encode all the words into their corresponding word embeddings. Word embeddings are vectors or 1d arrays. Each word has a unique embedding. Neural network only deals with numerical values; not strings or characters. That's why representing each word with an embedding is necessary. Then these embeddings are fed into an LSTM cell one by one. The hidden state of the previous cell state is fed back again in the cell by a feedback loop. The output of the drawn from the last cell state.

<img src="{{site.url}}{{site.baseurl}}/assets/images/figures/lstm.png" alt="a figure of LSTM">

Same process can be used to process videos beacause videos are basically sequence of images or frames. So in tasks like `Action Recognition` or `Video Captioning` LSTM architectures can be utilized. This type of architecture are called CNN-LSTMs.

Vanilla LSTMs work with 1-dimensional embeddings or features. A pretrained CNN can work as a good feature extractor. Each frame of the video is passed into CNN and the values on Dense Layer before the final Classification Layer can be taken as features. They are then passed inside LSTMs to learn sequence representations. 

Here is a diagram on a typical CNN.

<img src="{{site.url}}{{site.baseurl}}/assets/images/figures/CNN.png" alt="a figure of CNN">

Here is a diagram of basic CNN-LSTM architecture.

<img src="{{site.url}}{{site.baseurl}}/assets/images/figures/cnnlstm.png" alt="a figure of CNN-LSTM">

However, Vanilla 1d LSTM doesn't take into account spatial information. But we should exploit spatial features learned by CNNs for better spatio-temporal representation of video. For that reason instead of vanilla-LSTMs generally `convLSTM2D` is used. They take in 2d features instead of 1d and also output 2d features. In vanilla LSTM the inner gates or connections are connected by dense weights. But, in `convLSTM2D`, they are replaced with convolution operation. Because, connecting 2d layers with dense weights would cost huge number of parameters.This is same reason why in CNNs, the layers are connected by a convolution operation. 

<img src="{{site.url}}{{site.baseurl}}/assets/images/figures/cnnconvlstm.png" alt="a figure of CNN-convLSTM">

Here is basic `CNN-convLSTM` model in keras. Each video input has 32 frames. Each has a size of (224,224,3). So, input size is (32,224,224,3).

```python

x = Input(shape=(32,224,224,3))
cnn = MobileNet(include_top = false, input_shape = (224,224,3), weights = 'imagenet' )
x = TimeDistributed( cnn )(x)
x = ConvLSTM2D( filters = 256, return_sequence = False)(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(number_of_classes, activation='softmax')(x)
model = Model(inputs=[x], outputs=[predictions])

```

I hope in this article, you get a basic idea of how `CNN-LSTM` architectures work. To get more deeper understanding do visit the following links and try to play with them yourself.

You'll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

```ruby
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
```

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll's GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/