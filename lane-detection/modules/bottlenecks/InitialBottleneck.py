import torch
import torch.nn as nn

class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, relu=True):
        super().__init__()

        if relu: activation = nn.ReLU
        else: activation = nn.PReLU
        
        self.main_branch = nn.Conv2d(in_channels, out_channels - 1, kernel_size=3, stride=2, padding=1, bias=bias)

        # max pool & normalize
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # apply prelu after concat
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        out = torch.cat((main, ext), 1)
        out = self.batch_norm(out)
        return self.out_activation(out)
