
# 函数使用

## model..compile()
在 TensorFlow 中，Keras 模型的 .compile() 方法用于配置训练过程。以下是 .compile() 方法的一些关键参数：

optimizer (优化器):

作用： 设置优化算法，决定模型权重如何更新以最小化损失函数。
常见值： 'sgd' (随机梯度下降), 'adam' (Adam 优化器), 'rmsprop' (RMSprop 优化器) 等。
loss (损失函数):

作用： 定义训练过程中模型的误差，是优化的目标。
常见值： 'mean_squared_error' (均方误差), 'categorical_crossentropy' (分类交叉熵) 等。
metrics (评估指标):

作用： 用于监控训练和测试步骤。可以是一个字符串，也可以是一个字符串列表。
常见值： 'accuracy' (准确率), 'mae' (平均绝对误差) 等。
loss_weights (损失权重):

作用： 用于为不同的损失函数赋予不同的权重，以平衡它们对总体损失的贡献。
常见值： 字典，例如 {'output_1': 1.0, 'output_2': 0.5}。
sample_weight_mode (样本权重模式):

作用： 如果模型有多个输出，该参数用于指定如何使用 sample_weights。
常见值： None, 'temporal', 'None'。
weighted_metrics (加权指标):

作用： 在评估时给予不同输出不同的权重。
常见值： 与 metrics 类似，例如 ['accuracy', 'mae']。
target_tensors (目标张量):

作用： 用于使用外部的目标张量进行模型的训练。
常见值： Tensor 或 Tensor 列表。
distribute (分布式训练策略):

作用： 指定分布式训练策略。
常见值： None, 'mirrored', 'tpu' 等。
... (其他参数):

还有其他一些参数，例如 steps_per_execution（在每次执行中使用的批次数），experimental_steps_per_execution（实验性的步骤执行参数），run_eagerly（是否在 Eager 模式下运行）等。

## tf.keras.callbacks.ModelCheckpoint
是 TensorFlow 中的一个回调函数，用于在训练期间保存模型的权重。具体来说，它在每个训练周期结束后检查模型的性能，并在性能改善时保存模型的权重。
filepath:

作用： 保存模型权重的文件路径。可以包含格式化字符串，例如 {epoch:02d}-{val_loss:.2f}.h5，它将使用当前训练轮数和验证集损失的值来为文件生成唯一的名称。
示例： filepath='model_weights.h5'
monitor:

作用： 要监测的性能指标，例如 'val_loss' 或 'val_accuracy'。
示例： monitor='val_loss'
save_best_only:

作用： 如果设置为 True，则只有在监测的性能指标有所改善时才会保存模型。
示例： save_best_only=True
save_weights_only:

作用： 如果设置为 True，则只保存模型的权重，而不保存整个模型。
示例： save_weights_only=True
mode:

作用： 设置监测性能指标的模式，例如 'auto'、'min' 或 'max'。如果设置为 'min'，则保存监测指标的最小值时触发保存；如果设置为 'max'，
则保存监测指标的最大值时触发保存；如果设置为 'auto'，则根据监测指标的名称自动判断。
示例： mode='min'
save_freq:

作用： 定义保存频率。可以是 'epoch' 表示每个训练周期保存一次，或者整数表示每多少批次保存一次。
示例： save_freq='epoch' 或 save_freq=100

## model.fit()
其中各参数的含义如下：

x: 输入数据。可以是NumPy数组、Pandas DataFrame，也可以是 TensorFlow 的 Dataset 对象等。

y: 标签，即对应于输入数据的目标值。它应该与输入数据对应，用于训练模型。

epochs: 表示训练过程中数据将被迭代多少次。一个 epoch 表示将所有训练数据完整过一遍。

batch_size: 表示每次模型更新时使用的样本数。这对于大规模数据集是很重要的，因为不可能一次性将所有数据加载到内存中。

validation_data: 用于验证模型性能的数据集，可以是验证数据的输入和标签。

model.fit() 在训练过程中执行以下操作：

数据输入: 将输入数据和标签传递给模型。
前向传播: 使用当前模型参数对输入进行前向传播，生成预测。
计算损失: 计算模型输出与真实标签之间的损失。
反向传播: 根据损失计算梯度，并使用梯度下降等优化算法来更新模型参数。
迭代: 重复上述步骤，直到达到指定的 epoch 数量。
在训练过程中，fit() 方法还会输出一些训练信息，如每个 epoch 的损失值、验证损失值等。这些信息有助于了解模型的性能和收敛情况。

train_dataloader: 训练数据集，可以是 NumPy 数组、Pandas DataFrame，也可以是 TensorFlow 的 Dataset 对象等。这是用于模型训练的输入数据。

steps_per_epoch: 每个 epoch 中迭代的步数。通常是训练集样本数除以 batch_size。

epochs: 表示训练过程中数据将被迭代多少次。一个 epoch 表示将所有训练数据完整过一遍。

initial_epoch: 开始训练的初始 epoch 数，通常用于继续之前的训练。

validation_data: 用于验证模型性能的数据集，可以是验证数据的输入和标签。

validation_steps: 在每个 epoch 结束时从验证集中抽取的步数，通常是验证集样本数除以 batch_size。

callbacks: 一系列的回调函数，这些函数在训练的不同阶段会被调用。例如，用于保存模型、可视化训练过程等。

这个函数的作用是训练模型，执行以下操作：

数据输入: 将输入数据和标签传递给模型。

前向传播: 使用当前模型参数对输入进行前向传播，生成预测。

计算损失: 计算模型输出与真实标签之间的损失。

反向传播: 根据损失计算梯度，并使用梯度下降等优化算法来更新模型参数。

迭代: 重复上述步骤，直到达到指定的 epoch 数量。

验证: 在每个 epoch 结束时，使用验证数据集评估模型的性能。

回调函数: 在训练过程中调用回调函数，执行一些自定义操作，例如保存模型、可视化等。


## 理解 loss={‘yolo_loss’: lambda y_true, y_pred: y_pred})
这段代码是使用 Python 中的 lambda 函数来定义一个自定义的损失函数。在这里，损失函数被命名为 yolo_loss，它是一个函数，接受两个参数 y_true 和 y_pred，并返回 y_pred。

在深度学习中，损失函数是用来度量模型输出与实际目标之间的差异的函数。这个差异通常表示为模型对训练数据的预测与实际标签之间的误差。在这个特定的情境中，
使用了 lambda 函数创建了一个损失函数，但具体的计算逻辑在 y_pred 中完成。

这样的构造通常在自定义损失函数时使用，因为它允许以一种更灵活的方式定义损失函数，而不受传统函数定义的限制。

举例说明，这里的损失函数 yolo_loss 可能包含了一些复杂的逻辑，比如处理目标检测任务中的边界框坐标预测、类别预测等。然而，具体的计算逻辑需要在 y_pred 中完成。

下面是对输出中的一些关键信息的解释：

Layer (type): 层的类型，通常是网络中的一种层类型，例如 Dense、Conv2D 等。

Output Shape: 层的输出形状，表示该层输出的张量的形状。对于顺序模型（Sequential Model），每一层的输出形状都会显示在这里。

Param #: 层的可训练参数数量，表示该层的权重和偏置等可训练参数的总数。

Connected to: 显示与该层相连的输入张量或层。这有助于理解网络的连接结构。