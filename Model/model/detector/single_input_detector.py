import torch.nn as nn
class OneHeadDetector(nn.Module):
    def __init__(self, input_head, body, output_head):
        super().__init__()
        self.input_head = input_head
        self.body = body
        self.output_head = output_head

    def forward(self,data):
        data = self.input_head(data)
        data = self.body(data)
        data = self.output_head(data)
        return data