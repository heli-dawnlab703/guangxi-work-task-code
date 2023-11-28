import tensorflow as tf
from datetime import datetime
import os

# 定义模型
model = ...

# 定义数据加载器 train_dataloader 和 valid_dataloader
train_dataloader = ...
valid_dataloader = ...

# 定义训练轮数和初始轮数
epochs = 5
initial_epoch = 0

# 计算每轮的步数
train_steps_per_epoch = max(1, len(train_dataloader))
valid_steps_per_epoch = max(1, len(valid_dataloader))

# 定义回调函数列表 cbs
cbs = [...]

# 设置模型保存路径
modelcheckpoint_path = "./checkpoints/model_{epoch:02d}_{val_loss:.2f}.h5"
os.makedirs(os.path.dirname(modelcheckpoint_path), exist_ok=True)

# 设置 ModelCheckpoint 回调
modelcheckpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=modelcheckpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# 添加 ModelCheckpoint 回调到回调函数列表
cbs.append(modelcheckpoint_callback)

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(
    train_dataloader,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    initial_epoch=initial_epoch,
    validation_data=valid_dataloader,
    validation_steps=valid_steps_per_epoch,
    callbacks=cbs
)
