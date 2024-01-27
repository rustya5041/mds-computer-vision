import torch
import torch.nn as nn

class RegularBottleneck(nn.Module):
    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=0, dilation=1, asymmetric=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        assert internal_ratio > 1
        assert internal_ratio <= channels

        internal_channels = channels // internal_ratio

        if relu: activation = nn.ReLU
        else: activation = nn.PReLU

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d( channels, internal_channels, kernel_size=1, stride=1, bias=bias), 
            nn.BatchNorm2d(internal_channels), activation()
            )

        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), dilation=dilation, bias=bias), 
                nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(internal_channels,internal_channels,kernel_size=(1, kernel_size),stride=1,padding=(0, padding),dilation=dilation,bias=bias), 
                nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias), 
                nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, bias=bias), 
            nn.BatchNorm2d(channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(main + ext)