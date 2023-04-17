import torch.nn as nn

class DHMNeck(nn.Module):
    def __init__(self, d_model, sequence_length, neck_output):
        super(DHMNeck, self).__init__()
        self.mlp = nn.Linear(d_model * sequence_length, neck_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        x = self.relu(x)
        return x

class DHMClassesHead(nn.Module):
    def __init__(self, neck_output, hidden_size, bin_number):
        super(DHMClassesHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(neck_output, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_size, bin_number*bin_number)

    def forward(self, x):
        x = self.mlp(x)
        x = self.classifier(x)
        return x


class DHMRegressionHead(nn.Module):
    def __init__(self, neck_output, hidden_size):
        super(DHMRegressionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(neck_output, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.Regression = nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        x = self.Regression(x)
        x = self.sigmoid(x)
        return x


class transformer_encoder_v2(nn.Module):
    def __init__(self, transformer_encoder_layer_nums, transformer_encoder, neck_head, cls_head):
        super(transformer_encoder_v2, self).__init__()
        self.transformer_encoder = nn.ModuleList([nn.TransformerEncoderLayer(**transformer_encoder) for i in range(transformer_encoder_layer_nums)])
        self.neck_head = DHMNeck(**neck_head)
        self.cls_head = DHMClassesHead(**cls_head)

    def forward(self, src):
        for layer in self.transformer_encoder:
            src = layer(src)
        re = self.neck_head(src)
        class_result = self.cls_head(re)
        return class_result