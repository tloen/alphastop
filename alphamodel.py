import tensorflow as tf
import numpy as np
import nltk
import random
from nltk.corpus import words

AVG_WORD_LENGTH = 8
MAX_WORD_LENGTH = 19 
MAX_LENGTH = MAX_WORD_LENGTH + 1
HIDDEN_DIMENSION = 5
K = 10

def index(c):
  if c.islower():
    return ord(c) - ord('a') + 0
  else:
    return 26

def one_hot(c):
  v = np.zeros((27,), dtype='float32')
  v[index(c)] = 1
  return v

def encode_word(s):
  # reverse reading is better
  coded = np.array([one_hot(c) for c in s.lower()[::-1] + '_'])
  return np.resize(coded, (MAX_LENGTH, 27))

def random_letter():
  o = random.randint(0, 25)
  return chr(o + ord('a'))

word_list = words.words()

def generate_word(natural = False):
  if natural:
    s = random.choice(word_list)
    while len(s) > MAX_LENGTH:
      s = random.choice(word_list)
  else:
    s = ''
    for i in range(MAX_WORD_LENGTH):
      s += random_letter()
      if random.random() * AVG_WORD_LENGTH < 1:
        break
  return s

# generates an input matrix of shape (n, MAX_LENGTH, 27)
def generate_word_matrix(n):
  raws = []
  words = np.zeros((0, MAX_LENGTH, 27))
  kvp = []
  for i in range(n):
    raw = generate_word(natural = True)
    raws.append(raw)
  raws.sort()
  for raw in raws:
    word = encode_word(raw)
    words = np.vstack([words, word.reshape(1, MAX_LENGTH, 27)])
  return words, raws

def sentence_to_words(sentence):
  return sentence.split()

data = tf.placeholder(
  dtype=tf.float64,
  shape=(None, MAX_LENGTH, 27),
  name='input_matrix'
)

cell = tf.contrib.rnn.GRUCell(HIDDEN_DIMENSION, name="gru_cell")

# outputs is a tensor of shape (n, MAX_LENGTH, hidden_state)
outputs, state = tf.nn.dynamic_rnn(
  cell=cell,
  inputs=data,
  dtype=tf.float64
)

scores = tf.exp(outputs[:, -1, 0])

lognum = tf.reduce_sum(
  scores[:K]
)

logdenom = tf.constant(0.0, dtype=tf.float64)
for i in range(K):
  logdenom = tf.add(logdenom, tf.log(tf.reduce_sum(scores[i:])))

negative_log_probability = tf.negative(tf.subtract(lognum, logdenom))
opt = tf.train.AdamOptimizer(0.0001)
train = opt.minimize(negative_log_probability)