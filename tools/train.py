from Model import MyGNN, Runner, Logger
import torch.nn as nn
import torch

model = MyGNN(8, 256, 2, 3)
loss = nn.L1Loss()
opt = torch.optim.AdamW(model.parameters(), lr=0.3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = Logger('checkpoints', 'basic_gnn')

runner = Runner(
    model=model,
    loss=loss,
    optimzer=opt,
    device=device,
    logger=logger,
    max_epoch=10,
    batchsize=96
)

runner.run()