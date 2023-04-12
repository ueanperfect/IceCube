import torch
import torch.nn as nn
import math
import torch.nn.init as init

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward,output_features):
        super(TransformerModel, self).__init__()
        # self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dim_feedforward)
        self.fc = nn.Linear(d_model*128, output_features)
        self.softmax = nn.functional.softmax
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        # src = self.pos_encoder(src)
        output = self.transformer(src, src, src_mask=src_mask)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        output = self.softmax(output,dim=1)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model - 1, 2).float() * (-math.log(10000.0) / d_model))
        pe[:,:-1][:, 0::2] = torch.sin(position * div_term)
        pe[:,:-1][:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
