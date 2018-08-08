import tensorflow as tf


x = [1,2,3]
y = [1,2,3]

w = tf.Variable(5.0)

hypothesis = x*w

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(w))
    sess.run(train)
