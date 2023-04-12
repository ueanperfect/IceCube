import torch.nn as nn

class TwoHeadDetector(nn.Module):
    def __init__(self, input_head1, input_head2, body, output_head):
        super().__init__()
        self.input_head1 = input_head1
        self.input_head2 = input_head2
        self.body = body
        self.output_head = output_head

    def forward(self, data1,data2):
        data1 = self.input_head1(data1)
        data2 = self.input_head2(data2)
        data = self.body(data1,data2)
        data = self.output_head(data)
        return data