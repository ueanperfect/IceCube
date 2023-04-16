import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        mask = create_padding_mask(x,0)
        query = self.query(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_scores = attention_scores.masked_fill_(mask,-1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.linear(context)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_self_attention = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_output = self.multi_head_self_attention(x)
        x = self.norm1(x + attention_output)
        ff_output = self.feed_forward(x)
        return self.norm2(x + ff_output)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.relu = nn.functional.relu
        self.softmax = nn.functional.softmax
        self.mlp_1 = nn.Linear(d_model*128,4096,bias=True)
        self.mlp_2 = nn.Linear(4096,144)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.mlp_1(x)
        x = self.relu(x)
        x = self.mlp_2(x)
        return x

def create_padding_mask(src, pad_token_id):
    mask = (src == pad_token_id).any(dim=-1).unsqueeze(1).unsqueeze(2)
    return mask



# # 随机生成一批输入数据
# inputs = torch.randn(batch_size, 128, d_model)
#
# # 将数据输入到 Transformer Encoder 并获得输出
# encoder_outputs = encoder(inputs)
#
# # 输出形状应该与输入形状相同
# print(encoder_outputs.shape)  # 应为 (batch_size, seq_length, d_model)

