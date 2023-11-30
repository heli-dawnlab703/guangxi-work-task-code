from tensorflow.keras import layers, models
import tensorflow as tf


# 定义 FirstBlock
class FirstBlock(layers.Layer):
    def __init__(self, base_channels):
        super(FirstBlock, self).__init__()
        self.conv_for_feat3 = layers.Conv2D(int(base_channels * 8), (1, 1), padding='same')
        self.bn = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU(alpha=0.1)

    def call(self, inputs):
        feat3, feat2 = inputs
        x = self.conv_for_feat3(feat3)
        x = self.bn(x)
        x = self.leaky_relu(x)
        P5 = layers.concatenate([layers.UpSampling2D()(x), feat2], axis=-1)
        return P5


# 定义 SecondBlock
class SecondBlock(layers.Layer):
    def __init__(self):
        super(SecondBlock, self).__init__()
        self.add_layer = layers.Add()

    def call(self, inputs):
        P5_upsample_feat2_concat, x = inputs
        output = self.add_layer([P5_upsample_feat2_concat, x])
        return output


base_channels = 64  # 假设 base_channels 为 64
# input_shape_feat3 = tf.keras.layers.Input(shape=(None, None, int(base_channels * 4)))
# feat2 = tf.keras.layers.Input(shape=(None, None, int(base_channels * 8)))

# 使用 Sequential 构建模型并调用 build 方法
model = models.Sequential([
    FirstBlock(base_channels),
    SecondBlock()
])

# 如果你不确定输入形状，可以在创建模型时不提供输入形状，而是等到训练时根据输入数据的形状自动推断
# build 方法会根据输入形状自动构建模型
