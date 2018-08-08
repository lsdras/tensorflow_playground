import tensorflow as tf
tf.set_random_seed(777)

filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the decoded result.
record_defaults = [[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([3,1], name='weight'))
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_batch, y:y_batch})
    if step % 10 ==0:
        print(step, "cost: ", cost_val, "\nPredixtion:\n",hy_val)


coord.request_stop()
coord.join(threads)

print("Your score will be ", sess.run(hypothesis, feed_dict={x: [[100, 70, 101]]}))

print("Other scores will be ", sess.run(hypothesis,feed_dict={x: [[60, 70, 110], [90, 100, 80]]}))
