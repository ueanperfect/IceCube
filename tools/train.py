from Model import MyGNN, RunnerOriginal, Logger,basic_loss,RunnerLSTM,TransformerModel,TransformerEncoder
import torch.nn as nn
import torch

# 定义参数
d_model = 8  # 模型维度
num_heads = 2  # 多头注意力的头数
d_ff = 1024  # 前馈神经网络的隐藏层维度
num_layers = 6  # Transformer Encoder层数
seq_length = 50  # 序列长度
batch_size = 32  # 批量大小

# 创建一个 Transformer Encoder 实例
model = TransformerEncoder(d_model, num_heads, d_ff, num_layers)

opt = torch.optim.AdamW(model.parameters(), lr=0.00001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss = nn.CrossEntropyLoss().to(device)
model.to(device)
logger = Logger('checkpoints', 'transformer_encoder')

runner = RunnerLSTM(
    model=model,
    loss=loss,
    optimzer=opt,
    device=device,
    logger=logger,
    batch_ids_s=[101,102,103,104,105,400,401,402,403,404],
    max_epoch=1000,
    batchsize=2048
)
runner.run()