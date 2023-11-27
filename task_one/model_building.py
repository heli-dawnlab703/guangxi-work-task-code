import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, Concatenate


def DarknetConv2D_BN_Leaky(inputs, filters, kernel_size, name):
    x = Conv2D(filters, kernel_size, padding='same', name=name + '_conv')(inputs)
    x = BatchNormalization(name=name + '_bn')(x)
    x = LeakyReLU(alpha=0.1, name=name + '_leaky')(x)
    return x


def build_structure_block(feat3, feat2, base_channels):
    # DarknetConv2D_BN_Leaky 层
    conv_for_feat3 = DarknetConv2D_BN_Leaky(feat3, int(base_channels * 8), (1, 1), 'conv_for_feat3')
    P5 = conv_for_feat3

    # 上采样层
    P5_upsample = UpSampling2D()(P5)

    # 拼接层
    P5_upsample = Concatenate(axis=-1)([P5_upsample, feat2])

    return P5_upsample


# 测试
feat3 = tf.keras.layers.Input(shape=(None, None, 256))  # 输入形状需要根据实际情况调整
feat2 = tf.keras.layers.Input(shape=(None, None, 128))  # 输入形状需要根据实际情况调整
base_channels = 32  # 根据实际情况调整

output = build_structure_block(feat3, feat2, base_channels)

# 创建模型
model = tf.keras.Model(inputs=[feat3, feat2], outputs=output)

# 打印模型结构
model.summary()

