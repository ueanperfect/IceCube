from Model import Logger,RunnerClassification,TransformerEncoderV1
import torch.nn as nn
import torch

# 定义参数
d_model = 8  # 模型维度
num_heads = 2  # 多头注意力的头数
d_ff = 1024  # 前馈神经网络的隐藏层维度
num_layers = 8  # Transformer Encoder层数

# 创建一个 Transformer Encoder 实例
model = TransformerEncoderV1(d_model, num_heads, d_ff, num_layers,12)
opt = torch.optim.AdamW(model.parameters(), lr=0.000001,weight_decay=0.0001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss = nn.CrossEntropyLoss().to(device)
model.to(device)
logger = Logger('checkpoints', 'transformer_encoder_v1')

runner = RunnerClassification(
    model=model,
    loss=loss,
    optimzer=opt,
    device=device,
    logger=logger,
    batch_ids_s=[[101,102,103,104,105,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,512,513,514,515,516,517,518]],
    # batch_ids_s=[[513,514,515,516,517,518,101,102,103,105,414,415,416,417,418,419,512,104]],
    # batch_ids_s=[[101]],
    max_epoch=4000,
    batchsize=2880,
    bin_number=12,
    resume="/home/IceCube/checkpoints/transformer_encoderv2/2023-04-17_09-47-55/checkpoints/transformer_encoderv2_1_56_.pth",
)

runner.run()