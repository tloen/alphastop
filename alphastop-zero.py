import tensorflow as tf
import alphamodel as am

cost = am.negative_log_probability
train = am.train
init = tf.global_variables_initializer()
data = am.data

with tf.Session() as sess:
  def attempt_sort(n):
    words, raws = am.generate_word_matrix(5)
    scores = sess.run(am.scores, feed_dict = {data: words})
    sw = list(zip(scores.tolist(), raws))
    sw.sort()
    ordered = [w for s, w in sw]
    return sw

  sess.run(init)
  moving_average_cost = 0.5
  words, raws = am.generate_word_matrix(5)
  for t in range(100000):
    words, raws = am.generate_word_matrix(5)
    c, _ = sess.run([cost, train], feed_dict={data: words})
    moving_average_cost = 0.99 * moving_average_cost + 0.01 * c
    if (t + 1) % 100 == 0:
      print(t + 1, attempt_sort(5))
  
    