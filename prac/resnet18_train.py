from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os
from resnet import resnet18

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

def preprocess(x, y):
    # 将数据映射到-1~1
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)  # 类型转换
    y = tf.one_hot(y, depth=10)
    return x, y


(x, y), (x_test, y_test) = datasets.cifar10.load_data()  # 加载数据集
y = tf.squeeze(y, axis=1)  # 删除不必要的维度
y_test = tf.squeeze(y_test, axis=1)  # 删除不必要的维度
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))  # 构建训练集
# 随机打散，预处理，批量化
train_dataloader = train_db.shuffle(1000).map(preprocess).batch(512)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))  # 构建测试集
# 随机打散，预处理，批量化
valid_dataloader = test_db.map(preprocess).batch(512)
# 采样一个样本
sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main():
    # 2) 模型构建
    # [b, 32, 32, 3] => [b, 1, 1, 512]
    model = resnet18()  # ResNet18网络
    # model.build(input_shape=(None, 32, 32, 3))
    # model.summary()  # 统计网络参数
    optimizer = optimizers.Adam(lr=1e-4)  # 构建优化器

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
    epochs = 5
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
    main()
