import tensorflow as tf

# 假设 model 是你的模型
model = ...

# 定义损失函数，这里使用 lambda 函数简化
Init_lr_fit = 0.01
# 设置优化器，使用 Adam，并设置学习率和 beta_1
optimizer = tf.keras.optimizers.Adam(learning_rate=Init_lr_fit, beta_1=0.37)

# 编译模型
model.compile(optimizer=optimizer, loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

# 注意：这里的损失函数和优化器参数应该根据你的实际需求和网络结构进行调整
