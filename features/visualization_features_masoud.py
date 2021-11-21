#coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from collections import Counter
from bidi.algorithm import get_display
import arabic_reshaper
import matplotlib.pyplot as plt
import seaborn as sns
import os



def plot_ngram(data, col="not_set_yet", n_gram=2, output_name='n_gram_plot', n_most=5):
  """
  this function will plot most reapeted ngrams for you
  parameters : 
    data : the data you send to get the ngram from, you can send list of strings, one string, or dataframe 
    but be aware if you send daataframe you must fill col param too

    col : name of the column you want to get ngram from

    n_gram = set a int number for n in n-gram
    output_name : name of the jpg file you will get

    n_most : how many of most repeated ngrams will be shown.
  """
  the_corpus = data
  if isinstance(data, str):
    the_corpus = [data]
  if isinstance(data, pd.DataFrame):
    if col == "not_set_yet" : assert print("please set which column you want to check")
    the_corpus = data[col]
  vec = CountVectorizer(ngram_range=(n_gram, n_gram)).fit(the_corpus)
  bag_of_words = vec.transform(the_corpus)
  sum_words = bag_of_words.sum(axis=0)
  words_freq = [(get_display(arabic_reshaper.reshape(word))  , sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
  # https://www.programiz.com/python-programming/methods/built-in/sorted ==>  sorted more info
  words_freq2 = sorted(words_freq, key = lambda x: x[1], reverse=True)
  sns.set(style='dark')
  g = sns.barplot(y="n-gram", x="count", data=pd.DataFrame(words_freq2[:n_most], columns=['n-gram','count']))
  plt.xticks(rotation=70)
  fig = g.get_figure()
  if not os.path.exists('./plots_images/'):
    os.makedirs('./plots_images/')
  # bbox_inches is for showing all the plot parts.
  fig.savefig(f'./plots_images/{output_name}.jpg' , bbox_inches="tight")
  return "plot is ready"

def plot_rare_words(data, col="not_set_yet", from_row=0, to_row=5, output_name='rare_words_plot'):
  """
  this function will plot most reapeted ngrams for you
  parameters : 
    data : the data you send to get the rarewords from, you can send one string, list of strings, or dataframe 
    but be aware if you send daataframe you must fill col param too

    col : name of the column you want to get ngram from

    from_row : the words are sorted by their occurrence. how many rare words you want to see?  [this row ,to number]
    to_row : the words are sorted by their occurrence. how many rare words you want to see? [this row ,to number]
    output_name : name of the jpg file you will get

  """
  the_corpus = data
  if isinstance(data, str):
    the_corpus = pd.DataFrame([data], columns=['text'])
    col = 'text'
  if isinstance(data, list):
    the_corpus = pd.DataFrame(data, columns=['text'])
    col = 'text'
  if isinstance(data, pd.DataFrame):
    if col == "not_set_yet" : assert print("please set which column you want to check")
    the_corpus = data

  rare_words = Counter([item for row in the_corpus[col] for item in row.split(" ")])
  rare_words_df = pd.DataFrame(rare_words.most_common())
  rare_words_df.columns = ['word', 'count']
  rare_words_df['word'] = rare_words_df.word.apply(lambda x : get_display(arabic_reshaper.reshape(x) ))
  sns.set(style='dark')
  g = sns.barplot(y="word", x="count", data=rare_words_df.sort_values(by=['count']).iloc[from_row:to_row])
  fig = g.get_figure()
  # bbox_inches is for showing all the plot parts.
  if not os.path.exists('./plots_images/'):
    os.makedirs('./plots_images/')
  # bbox_inches is for showing all the plot parts.
  fig.savefig(f'./plots_images/{output_name}.jpg' , bbox_inches="tight")
  return "plot is ready"
