import tensorflow as tf
tf.set_random_seed(777)

x_train = [1,2,3]
y_train = [1,2,3]
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])
w = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = x * w + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, w_val, b_val, _ = sess.run([cost, w, b, train], feed_dict={x:[1,2,3], y:[1,2,3]})
    if step%20 == 0:
        print(step, cost_val, w_val, b_val)
