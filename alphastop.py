import tensorflow as tf
import numpy as np
import random

def index(c):
  if c.islower():
    return ord(c) - ord('a') + 0
  else:
    return 26

def one_hot(c):
  v = np.zeros((27,))
  v[index(c)] = 1
  return v

max_length = 20

def encode_word(s):
  coded = np.array([one_hot(c) for c in s[::-1] + '_'])
  return np.resize(coded, (max_length, 27))

def random_letter():
  o = random.randint(0, 25)
  return chr(o + ord('a'))

def generate_word():
  s = ''
  while True:
    s += random_letter()
    if random.random() < 0.25 or len(s) == max_length - 10:
    #if len(s) == max_length - 1:
      break
  return encode_word(s), s

def generate_words(n):
  raws = []
  words = np.zeros((0, max_length, 27))
  for i in range(n):
    word, raw = generate_word()
    raws.append(raw)
    words = np.vstack([words, word.reshape(1, max_length, 27)])
  return words, raws

def generate_stub():
  s = ''
  while True:
    if random.random() < 0.6 or len(s) == 10:
      break
    s += random_letter()
  return s

def generate_two_words():
  stub = generate_stub()
  s1 = stub
  s2 = stub
  while True:
    s1 += random_letter()
    if random.random() < 0.25 or len(s1) == max_length:
    #if len(s) == max_length - 1:
      break
  while True:
    s2 += random_letter()
    if random.random() < 0.25 or len(s2) == max_length:
    #if len(s) == max_length - 1:
      break
  return np.stack((encode_word(s1), encode_word(s2))), [s1, s2]

def sentence_to_words(sentence):
  return sentence.split()

x_i = tf.placeholder(
  dtype=tf.float32, 
  shape=(max_length, 27)
)

x_j = tf.placeholder(
  dtype=tf.float32, 
  shape=(max_length, 27)
)

data = tf.placeholder_with_default(
  input=tf.stack((x_i, x_j)),
  shape=(None, max_length, 27)
)

print(data.shape)

hidden_dimension = 5

batch_size = tf.placeholder_with_default(
  input=tf.constant(2),
  shape=()
)

cell = tf.contrib.rnn.GRUCell(hidden_dimension, name="gru_cell")

outputs, state = tf.nn.dynamic_rnn(
  cell=cell,
  inputs=data,
  initial_state = cell.zero_state(
    batch_size,
    dtype = tf.float32
  )
)

sigma = 10.0

score_i = outputs[0][-1][0]
score_j = outputs[1][-1][0]
z = tf.multiply(-sigma, tf.subtract(score_i, score_j))
p_ij = tf.divide(1.0, tf.add(1.0, tf.exp(z)))
cost = tf.negative(tf.log(p_ij))
opt = tf.train.AdamOptimizer(0.0001)
train = opt.minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  moving_average_accuracy = 0.5
  for t in range(1000000):
    words, raws = generate_two_words()
    if raws[0] < raws[1]:
      i, j = 1, 0
    else:
      i, j = 0, 1
    c, _ = sess.run([cost, train], feed_dict={x_i: words[i], x_j: words[j]})

    moving_average_accuracy = 0.999 * moving_average_accuracy + 0.001 * (c < 0.69)
    if (t + 1) % 1000 == 0:
      print(t + 1, moving_average_accuracy)
  def word_score(word):
    return sess.run(outputs, 
    feed_dict={
      data: np.reshape(encode_word(word), (-1, max_length, 27)),
      batch_size: 1,
      x_i: encode_word(word),
      x_j: encode_word(word)
      })[0, -1, 0]
  def sort_sentence(sentence):
    words = sentence_to_words(sentence)
    pairs = [(word_score(w), w) for w in words]
    p2 = sorted(pairs)
    l = [w for s, w in p2]
    return l
  print(sort_sentence("well seymour i made it despite your directions"))
  print(sort_sentence("ah superintendent chalmers welcome i hope youre ready for an unforgettable luncheon"))
  saver = tf.train.Saver()
  save_path = saver.save(sess, "./alphastop_params.ckpt")
  print("Model saved in %s" % save_path)






