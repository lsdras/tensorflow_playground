import numpy as np
import tensorflow as tf

xy = np.genfromtxt('train.csv', delimiter=',',dtype='str')
for col in range(xy.shape[0]):
    for row in range(xy.shape[1]):
        if xy[col,row]=="":
            xy[col,row]="Empty"

x_data = xy[1:, 2:]
y_data = xy[1:, [1]]

nb_classes = 2  # Dead:0 / Alive:1


#Pclass(0),Sex(1)(m0 f1 empty0.5),Age(2)(linear reg),SibSp(3),Parch(4),Fare(5),Embarked(6)(2 empty)
x_data = np.delete(x_data,[1,2,7,9],1)
print(x_data.shape, y_data.shape)
print(x_data)


col = x_data.shape[0]
x = []

#Pclass encoding
for i in range(col):
   x.append(np.float32(x_data[i,0]))

#gender encoding
for i in range(col):
    sex = x_data[i,1]
    if sex == 'male':
        x.append(0)
    elif sex == 'female':
        x.append(1)

#age encoding/ linear reg for empty value is necessary(177/891 are empty)

avg_age = []
for i in range(col):
    age = x_data[i,2]
    if age != 'Empty': avg_age.append(float(age))
avg_age = sum(avg_age)/len(avg_age)

for i in range(col):
    age = x_data[i,2]
    if age != 'Empty': x.append(float(age))
    else: x.append(avg_age)
'''
for i in range(col):
    age = x_data[i,2]
    if age != 'Empty': x.append(float(age))
    else: x.append(-1)
'''
#SibSp
for i in range(col):
    x.append(np.float32(x_data[i,3]))
#Parch
for i in range(col):
    x.append(np.float32(x_data[i,4]))
#Fare
for i in range(col):
    x.append(np.float32(x_data[i,5]))

class1=[]
class2=[]
class3=[]
for i in range(col):
    if x_data[i,0]=='1':class1.append(np.float32(x_data[i,5]))
    elif x_data[i,0]=='2':class2.append(np.float32(x_data[i,5]))
    elif x_data[i,0]=='3':class3.append(np.float32(x_data[i,5]))
class1_avg_fare = sum(class1)/len(class1)
class2_avg_fare = sum(class2)/len(class2)
class3_avg_fare = sum(class3)/len(class3)

for i in range(col):
    embark = x_data[i,6]
    if embark == 'C': x.append(0)
    elif embark == 'Q': x.append(1)
    elif embark == 'S': x.append(2)
    else: x.append(3)

x = np.array(x)
x_data = x.reshape((-1,7),order='F')



X = tf.placeholder(tf.float32, [None, 7])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ,1
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape", Y_one_hot)


'''
W = tf.Variable(tf.random_normal([7, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
'''

W1 = tf.get_variable("W1", shape=[7, 8],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([8]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[8, 16],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([16]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[16, 8],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([8]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

W4 = tf.get_variable("W4", shape=[8, 2],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([2]))
logits = tf.matmul(L3, W4) + b4
hypothesis = tf.nn.softmax(logits)


# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
'''
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
'''
# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Launch graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10000):
    sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        loss, acc = sess.run([cost, accuracy], feed_dict={
                             X: x_data, Y: y_data})
        print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
            step, loss, acc))

# Let's see if we can predict
pred = sess.run(prediction, feed_dict={X: x_data})
# y_data: (N,1) = flatten => (N, ) matches pred.shape
for p, y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


print("~~~~~~~~~~~~~~~validation~~~~~~~~~~~~~~")

xy = np.genfromtxt('test.csv', delimiter=',',dtype='str')
for col in range(xy.shape[0]):
    for row in range(xy.shape[1]):
        if xy[col,row]=="":
            xy[col,row]="Empty"

x_data = xy[:, 1:]
y_data = xy[:, [0]]

nb_classes = 2  # Dead:0 / Alive:1


#Pclass(0),Sex(1)(m0 f1 empty0.5),Age(2)(linear reg),SibSp(3),Parch(4),Fare(5),Embarked(6)(2 empty)
x_data = np.delete(x_data,[1,2,7,9],1)
print(x_data.shape, y_data.shape)
print(x_data)


col = x_data.shape[0]
x = []
#Pclass encoding
for i in range(col):
   x.append(np.float32(x_data[i,0]))

#gender encoding
for i in range(col):
    sex = x_data[i,1]
    if sex == 'male':
        x.append(0)
    elif sex == 'female':
        x.append(1)

#age encoding/ linear reg for empty value is necessary(177/891 are empty)

for i in range(col):
    age = x_data[i,2]
    if age != 'Empty': x.append(float(age))
    else: x.append(avg_age)




#SibSp
for i in range(col):
    x.append(np.float32(x_data[i,3]))
#Parch
for i in range(col):
    x.append(np.float32(x_data[i,4]))
#Fare
for i in range(col):
    fare = x_data[i,5]
    if fare != 'Empty': x.append(np.float32(fare))
    else:
        if x_data[i,0]=='1':x.append(class1_avg_fare)
        elif x_data[i,0]=='2':x.append(class2_avg_fare)
        elif x_data[i,0]=='3':x.append(class3_avg_fare)

for i in range(col):
    embark = x_data[i,6]
    if embark == 'C': x.append(0)
    elif embark == 'Q': x.append(1)
    elif embark == 'S': x.append(2)
    else: x.append(3)

x = np.array(x)
x_data = x.reshape((-1,7),order='F')

pred = sess.run(prediction, feed_dict={X: x_data})
# y_data: (N,1) = flatten => (N, ) matches pred.shape
for p, y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {}".format(int(y), p))

import csv
f = open('output.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
for p, y in zip(pred, y_data.flatten()):
    wr.writerow([int(y), p])
f.close()
