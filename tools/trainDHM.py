from Model import DHM, Logger, RunnerDHM, DHMLoss, RunnerLSTM
import torch.nn as nn
import torch

DHMModel_parameter = dict(
    transformer_encoder_layer_nums=10,
    transformer_encoder=dict(
        d_model=8,
        nhead=2,
        dim_feedforward=1024,
        batch_first=True
    ),
    neck_head=dict(
        d_model=8,
        sequence_length=128,
        neck_output=2048,
    ),
    cls_head=dict(
        neck_output=2048,
        hidden_size=1024,
        output_class=10 * 10),
    regression_head=dict(
        neck_output=2048,
        hidden_size=1024
    )
)
model = DHM(**DHMModel_parameter)

opt = torch.optim.AdamW(model.parameters(), lr=0.000005, weight_decay=0.00001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss = torch.nn.CrossEntropyLoss()
model.to(device)
logger = Logger('checkpoints', 'DHM')

runner = RunnerLSTM(
    model=model,
    loss=loss,
    optimzer=opt,
    device=device,
    logger=logger,
    # batch_ids_s=[[101,102,103,104,105,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,512,513,514,515,516,517,518]],
    # batch_ids_s=[[513,514,515,516,517,518,101,102,103,105,414,415,416,417,418,419,512,104]],
    batch_ids_s=[[101]],
    max_epoch=4000,
    batchsize=1024,
)
runner.run()
