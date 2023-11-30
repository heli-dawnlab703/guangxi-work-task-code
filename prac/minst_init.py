import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y


# 1) 数据dataloader 进行构建
batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x,y))
train_dataloader = db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
valid_dataloader = ds_val.map(preprocess).batch(batchsz)

sample = next(iter(train_dataloader))
print(sample[0].shape, sample[1].shape)

db_iter = iter(db)
sample = next(db_iter)
print('batch:', sample[0].shape, sample[1].shape)


# 2) 模型构建
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

# optimizer = optimizers.SGD(learning_rate=0.001)
Init_lr_fit = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=Init_lr_fit, beta_1=0.37)

# 3)模型编译
model.compile(optimizer=optimizer,
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 4) 模型训练
# 4-1) 定义日志存储路径
log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# 创建日志目录
os.makedirs(log_dir, exist_ok=True)
# 设置 TensorBoard 回调
'''
log_dir： 日志目录的路径，这是存储TensorBoard事件文件的位置。TensorBoard将在这个目录中创建事件文件，以便后续的可视化。
histogram_freq： 控制直方图计算的频率。如果设置为1，则每个训练周期都会计算一次直方图。直方图可用于查看权重分布的变化。
'''
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# 4-2) 设置保存模型
# 定义保存模型的路径
modelcheckpoint_path = "./checkpoints/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/model_weights.h5"
# 创建保存模型的目录
os.makedirs(os.path.dirname(modelcheckpoint_path), exist_ok=True)
# 设置 ModelCheckpoint 回调
modelcheckpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=modelcheckpoint_path,
    monitor='val_loss',  # 监控验证集损失
    save_best_only=True,  # 只保存最好的模型
    save_weights_only=True  # 保存整个模型，包括权重和模型结构
)
cbs = [tensorboard_callback, modelcheckpoint_callback]
# 定义训练轮数和初始轮数
epochs = 6
initial_epoch = 0
# 计算每轮的步数
train_steps_per_epoch = max(1, len(train_dataloader))
valid_steps_per_epoch = max(1, len(valid_dataloader))

# 4-3) 训练模型
model.fit(
    train_dataloader,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    initial_epoch=initial_epoch,
    validation_data=valid_dataloader,
    validation_steps=valid_steps_per_epoch,
    callbacks=cbs
)


if __name__ == '__main__':
    pass
