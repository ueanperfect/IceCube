from Model import *
import torch.nn as nn
import torch

model = MyGNN(8, 16, 2, 3)
loss = nn.L1Loss()
opt = torch.optim.AdamW(model.parameters(), lr=0.3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tranner = basic_gnn_trainner(
    model=model,
    loss=loss,
    optimzer=opt,
    epochs=10,
    batchsize=128,
    device=device,
)

tranner.run()