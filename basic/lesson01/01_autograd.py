import tensorflow as tf


x = tf.constant(1.)
a = tf.constant(2.)
b = tf.constant(3.)
c = tf.constant(4.)

'''
在 TensorFlow 中，tf.GradientTape 是一种上下文管理器，用于记录计算过程中涉及的所有操作，以便后续计算梯度。
在训练神经网络时，通常使用梯度下降等优化算法来调整模型参数，而梯度的计算则是通过反向传播来实现的。

tape.watch(x) 用于告诉 TensorFlow 跟踪对 x 的操作。
然后，在 tape 上下文内执行前向计算，最后使用 tape.gradient 计算相对于 x 的梯度。

'''
with tf.GradientTape() as tape:
	tape.watch([a, b, c])
	y = a**2 * x + b * x + c


[dy_da, dy_db, dy_dc] = tape.gradient(y, [a, b, c])
print(dy_da, dy_db, dy_dc)  # 输出 dy/da # 输出 dy/db # 输出 dy/dc