---
title:  "Using LSTMs for computer vision tasks"
header:
  teaser: "/assets/images/500x300.png"
categories: 
  - deep learning
tags:
  - lstm
  - computer vision
---

LSTM have always been widely successful in Natural Language Processin. It is particularly designed for processing sequence and language is a sequence of words.
Let's say we have a dataset of imdb reviews and we want to classify them into positive reviews and negative ones. First we encode all the words into their corresponding word embeddings. Word embeddings are vectors or 1d arrays. Each word has a unique embedding. Neural network only deals with numerical values; not strings or characters. That's why representing each word with an embedding is necessary. Then these embeddings are fed into an LSTM cell one by one. The hidden state of the previous cell state is fed back again in the cell by a feedback loop. The output of the drawn from the last cell state.

<img src="{{site.url}}{{site.baseurl}}/assets/images/figures/lstm.png" alt="a figure of LSTM">





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