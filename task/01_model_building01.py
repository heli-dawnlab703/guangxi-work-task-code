import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU

class DarknetConv2D_BN_Leaky(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, name):
        super(DarknetConv2D_BN_Leaky, self).__init__(name=name)
        self.conv = Conv2D(filters, kernel_size, strides=(1, 1), padding='same', use_bias=False)
        self.bn = BatchNormalization()
        self.leaky_relu = LeakyReLU(alpha=0.1)

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
        conv_feat3 = self.conv_for_feat3(feat3)
        upsampled_feat3 = self.upsample(conv_feat3)
        P5 = self.concatenate([upsampled_feat3, feat2])
        return P5

class SecondBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(SecondBlock, self).__init__()
        self.add_layer = tf.keras.layers.Add()

    def call(self, x, y):
        return self.add_layer([x, y])

# 示例用法
base_channels = 3  # 假设 base_channels 为 64
feat3 = tf.keras.layers.Input(shape=(None, None, int(base_channels * 4)))
feat2 = tf.keras.layers.Input(shape=(None, None, int(base_channels * 2)))

# 创建 DarknetConv2D_BN_Leaky 类实例
darknet_conv = DarknetConv2D_BN_Leaky(int(base_channels * 8), (1, 1), 'conv_for_feat3')
# 调用 DarknetConv2D_BN_Leaky 实例
conv_result = darknet_conv(feat3)

# 创建 FirstBlock 类实例
first_block = FirstBlock(base_channels)
# 调用 FirstBlock 实例
P5_result = first_block(feat3, feat2)

P5_result = layers.Dense(int(base_channels * 4), activation=None)(P5_result)
# 创建 SecondBlock 类实例
second_block = SecondBlock()
# 调用 SecondBlock 实例
output_result = second_block(feat3, P5_result)

# 创建模型
model = tf.keras.models.Model(inputs=[feat3, feat2], outputs=output_result)

# 输出模型概要
model.summary()
