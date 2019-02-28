import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.constant(1, shape=[100, 100, 100])
y = tf.constant(2, shape=[100, 100, 100])
z = x + y

print(z.eval())