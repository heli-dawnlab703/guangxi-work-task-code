import tensorflow as tf
from datetime import datetime
import os

# 假设 model 是你的模型
model = ...

# 定义保存模型的路径
modelcheckpoint_path = "./checkpoints/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/model_weights.h5"

# 创建保存模型的目录
os.makedirs(os.path.dirname(modelcheckpoint_path), exist_ok=True)

# 设置 ModelCheckpoint 回调
modelcheckpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=modelcheckpoint_path,
    monitor='val_loss',  # 监控验证集损失
    save_best_only=True,  # 只保存最好的模型
    save_weights_only=False,  # 保存整个模型，包括权重和模型结构
    mode='min',  # 当监测值为最小时触发保存
    verbose=1  # 显示保存信息
)
cbs = [modelcheckpoint_callback]