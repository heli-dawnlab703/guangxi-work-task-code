import tensorflow as tf

# 创建一个简单的神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(4,), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 定义一个损失函数
loss_fn = tf.keras.losses.BinaryCrossentropy()

# 定义一个优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 准备输入数据和目标值
input_data = tf.constant([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
target = tf.constant([[0.0], [1.0]])

# 定义一个前向传播的函数
def forward_pass(inputs):
    return model(inputs)

# 计算损失
with tf.GradientTape() as tape:
    predictions = forward_pass(input_data)
    loss = loss_fn(target, predictions)

# 计算梯度并应用反向传播
gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
