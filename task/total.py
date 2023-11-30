from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


# def preprocess(x, y):
#     """
#     x is a simple image, not a batch
#     """
#     x = tf.cast(x, dtype=tf.float32) / 255.
#     x = tf.reshape(x, [28 * 28])
#     y = tf.cast(y, dtype=tf.int32)
#     y = tf.one_hot(y, depth=10)
#     return x, y
#
#
# batchsz = 128
# (x, y), (x_val, y_val) = datasets.mnist.load_data()
# print('datasets:', x.shape, y.shape, x.min(), x.max())
#
# train_db = tf.data.Dataset.from_tensor_slices((x, y))
# train_db = train_db.map(preprocess).shuffle(60000).batch(batchsz)
# ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# ds_val = ds_val.map(preprocess).batch(batchsz)

class DarknetConv2D_BN_Leaky(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, name):
        super(DarknetConv2D_BN_Leaky, self).__init__(name=name)
        self.conv = Conv2D(filters, kernel_size, padding='same')
        self.bn = BatchNormalization()
        self.leaky_relu = LeakyReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class FirstBlock(tf.keras.layers.Layer):
    def __init__(self, base_channels):
        super(FirstBlock, self).__init__()
        self.conv_for_feat3 = DarknetConv2D_BN_Leaky(int(base_channels * 8), (1, 1), 'conv_for_feat3')
        self.upsample = tf.keras.layers.UpSampling2D()
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

    def call(self, feat3, feat2):
        P5 = self.conv_for_feat3(feat3)  #  int(base_channels * 8),
        P5_upsample = self.upsample(P5)
        P5 = self.concatenate([P5_upsample, feat2]) # 2 * int(base_channels * 8),
        return P5


class SecondBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(SecondBlock, self).__init__()
        self.add_layer = tf.keras.layers.Add()

    def call(self, x, y):
        return self.add_layer([x, y])



# 示例用法
base_channels = 1 # 假设 base_channels 为 64
feat3 = tf.keras.layers.Input(shape=(None, None, int(base_channels * 4)))
feat2 = tf.keras.layers.Input(shape=(None, None, int(base_channels * 8)))


# 2 * int(base_channels * 8)
# 创建 FirstBlock 类实例
first_block = FirstBlock(base_channels)
# 调用 FirstBlock 实例
P5_result = first_block(feat3, feat2)
layer1 = layers.Dense(int(base_channels * 4), activation=None)
P5_result = layer1(P5_result)
# 创建 SecondBlock 类实例
second_block = SecondBlock()

# 调用 SecondBlock 实例
output_result = second_block(feat3, P5_result)
# 创建模型
model = tf.keras.models.Model(inputs=[feat3, feat2], outputs=output_result)
# 输出模型概要
model.summary()

# 定义损失函数，这里使用 lambda 函数简化
Init_lr_fit = 0.01
# 设置优化器，使用 Adam，并设置学习率和 beta_1
'''
learning_rate（学习率）： 这是一个控制模型权重更新步长的超参数。它决定了在每次权重更新中应用的步长大小。较小的学习率可能导致模型收敛得更慢，但有助于获得更准确的权重。
通常，学习率是在训练过程中逐渐减小的。

beta_1： 这是Adam优化器的指数衰减率，控制了对梯度的一阶矩估计的衰减。它通常设置在0.9左右。较高的值会导致更快的衰减。
'''
optimizer = tf.keras.optimizers.Adam(learning_rate=Init_lr_fit, beta_1=0.37)

# 编译模型
model.compile(optimizer=optimizer, loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

# 定义日志存储路径
log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# 创建日志目录
os.makedirs(log_dir, exist_ok=True)
# 设置 TensorBoard 回调
'''
log_dir： 日志目录的路径，这是存储TensorBoard事件文件的位置。TensorBoard将在这个目录中创建事件文件，以便后续的可视化。

histogram_freq： 控制直方图计算的频率。如果设置为1，则每个训练周期都会计算一次直方图。直方图可用于查看权重分布的变化。

'''
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

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


# 定义数据加载器 train_dataloader 和 valid_dataloader
train_dataloader = ...
valid_dataloader = ...

# 定义训练轮数和初始轮数
epochs = 5
initial_epoch = 0

# 计算每轮的步数
train_steps_per_epoch = max(1, len(train_dataloader))
valid_steps_per_epoch = max(1, len(valid_dataloader))

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

#
# import tensorflow as tf
# height = 28
# width = 28
# channels = 3
# # 定义一个输入张量
# input_tensor = tf.keras.Input(shape=(height, width, channels))
#
# # 添加MaxPooling2D下采样层
# max_pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
# max_pooled_tensor = max_pooling_layer(input_tensor)
#
# # 添加AveragePooling2D下采样层
# average_pooling_layer = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
# average_pooled_tensor = average_pooling_layer(input_tensor)
#
# import tensorflow as tf
# from tensorflow.keras.layers import Input, MaxPooling2D, AveragePooling2D
#
# # 输入数据的形状，假设是 (batch_size, height, width, channels)
# input_shape = (None, 28, 28, 256)
#
# # 创建输入层
# inputs = Input(shape=input_shape[1:])
#
# # 下采样层使用最大池化
# max_pooling_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(inputs)
#
# # 下采样层使用平均池化
# average_pooling_layer = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(inputs)
#
# # 创建模型
# model_max_pooling = tf.keras.Model(inputs=inputs, outputs=max_pooling_layer)
# model_average_pooling = tf.keras.Model(inputs=inputs, outputs=average_pooling_layer)
#
# # 打印模型的架构
# model_max_pooling.summary()
# model_average_pooling.summary()
