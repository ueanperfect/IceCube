from Model import MyGNN, RunnerOriginal, Logger,basic_loss,RunnerLSTM,TransformerModel
import torch.nn as nn
import torch

# model = MyGNN(8, 512, 256, 8)

d_model = 8  # 嵌入和Transformer层的维度
nhead = 2  # 多头注意力的头数
num_layers = 5  # Transformer层的数量
dim_feedforward = 512  # 前馈神经网络的隐藏层大小

model = TransformerModel(d_model, nhead, num_layers, dim_feedforward,output_features=256)

opt = torch.optim.Adam(model.parameters(), lr=0.0001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss = nn.CrossEntropyLoss().to(device)
model.to(device)
logger = Logger('checkpoints', 'basic_gnn')

runner = RunnerLSTM(
    model=model,
    loss=loss,
    optimzer=opt,
    device=device,
    logger=logger,
    batch_ids_s=[101,102,103,104,105],
    max_epoch=10,
    batchsize=8
)
runner.run()