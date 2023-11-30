import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, Concatenate, Add

class DarknetConv2D_BN_Leaky(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, name):
        super(DarknetConv2D_BN_Leaky, self).__init__(name=name)
        self.conv = Conv2D(filters, kernel_size, strides=(1, 1), padding='same', use_bias=False, name=name + '_conv')
        self.bn = BatchNormalization(name=name + '_bn')
        self.leaky_relu = LeakyReLU(alpha=0.1, name=name + '_leaky')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

def build_first_block(base_channels):
    model = Sequential([
        DarknetConv2D_BN_Leaky(int(base_channels * 8), (1, 1), 'conv_for_feat3'),
        UpSampling2D(),
    ], name='first_block')
    return model

def build_second_block():
    model = Sequential([
        Add(),
    ], name='second_block')
    return model

# 创建模型
base_channels = 64
feat3 = tf.keras.Input(shape=(None, None, int(base_channels * 4)), name='feat3')
feat2 = tf.keras.Input(shape=(None, None, int(base_channels * 4)), name='feat2')

first_block = build_first_block(base_channels)
P5_result = first_block(feat3)

P5_result = layers.Dense(int(base_channels * 4), activation=None)(P5_result)
second_block = build_second_block()
y = second_block([feat3, P5_result])
# 创建包含输入和输出的模型
model = tf.keras.Model(inputs=[feat3, feat2], outputs=y)

# 打印模型信息
model.summary()
