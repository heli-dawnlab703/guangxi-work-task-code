import tensorflow as tf
from datetime import datetime
import os

# 假设 model 是你的模型
model = ...

# 定义日志存储路径
log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# 创建日志目录
os.makedirs(log_dir, exist_ok=True)

# 设置 TensorBoard 回调
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 编译模型时添加 TensorBoard 回调
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], callbacks=[tensorboard_callback])
