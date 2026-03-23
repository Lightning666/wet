# STM32 LSTM TensorFlow 项目

这是一个适用于 **STM32 微控制器部署场景** 的 TensorFlow 2.x / Keras LSTM 示例项目。

## 项目目标

该项目使用 8 维水质时间序列输入，构建一个参数量较小、结构简单、便于后续通过 **X-CUBE-AI** 导入的 LSTM 模型。

- 输入形状：`(64, 8)`
- 两层 LSTM：`32 -> 16`
- 一个小型全连接层：`16`
- 输出层：`softmax` 分类输出
- 总参数量远低于约 `50,000` 参数（约 `200 KB float32`）预算

## 目录结构

```text
wet/
├── README.md
├── requirements.txt
└── src/
    └── train_lstm_stm32.py
```

## 在 PyCharm 中运行

1. 用 PyCharm 打开本项目根目录。
2. 创建或选择 Python 解释器（建议 Python 3.10~3.12）。
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
4. 运行脚本：
   ```bash
   python src/train_lstm_stm32.py
   ```

## 运行结果

脚本会完成以下操作：

1. 构建符合 STM32 约束的 LSTM 网络。
2. 打印 `model.summary()`。
3. 打印总参数量和参数占用 KB。
4. 用随机数据进行少量 epoch 训练。
5. 保存模型到：
   - `lstm_stm32_model.h5`

## 为什么适合 STM32

- 仅使用 **2 个标准 LSTM 层**，避免复杂算子。
- 每层单元数较小，参数量低。
- 不使用 `BatchNormalization`、`Embedding`、自定义 RNN、复杂激活等不利于嵌入式部署的结构。
- 使用 `float32` 训练，但通过较小模型、轻微正则化和梯度裁剪，帮助后续做 8-bit 量化。
- 推理阶段可固定 `batch_size = 1`，适合 MCU 在线推理。

## 回归任务修改方式

如果你需要做回归而不是分类，可以修改 `src/train_lstm_stm32.py` 中最后一层与损失函数：

- 输出层改为：`Dense(1, activation="linear")`
- 损失函数改为：`mse` 或 `mae`

然后重新训练并保存即可。
