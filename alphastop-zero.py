import tensorflow as tf
import alphamodel as am
import numpy as np

cost = am.cost
train = am.train
init = tf.global_variables_initializer()
data = am.data
scores = am.scores

with tf.Session() as sess:
  def attempt_sort(n): 
    words, raws = am.generate_word_matrix(n)
    scores = sess.run(am.scores, feed_dict = {data: words})
    sw = list(zip(scores.tolist(), raws))
    sw.sort()
    # ordered = [w for s, w in sw]
    return sw
  def is_sorted(x):
    return all(sorted(x) == x)
  
  train_writer = tf.summary.FileWriter('./summaries/train_log_2', sess.graph)

  sess.run(init)
  words, raws = am.generate_word_matrix(2)
  moving_average_cost = sess.run(cost, feed_dict={data: words})
  moving_average_accuracy = 0.5
  for t in range(100000):
    words, raws = am.generate_word_matrix(2)
    m, c, _, s = sess.run([am.merged, cost, train, scores], feed_dict={data: words})
    if t % 100 == 0:
      train_writer.add_summary(m, t)

    moving_average_cost = 0.999 * moving_average_cost + 0.001 * c

    moving_average_accuracy = 0.999 * moving_average_accuracy + \
      0.001 * is_sorted(s)
    if (t + 1) % 100 == 0:
      print(t + 1, np.exp(-moving_average_cost))