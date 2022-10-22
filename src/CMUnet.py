import torch
import torch.nn as nn
import torch.nn.functional as F
from src.msag import MSAG

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerBlock(nn.Module):
    def __init__(self, dim, depth):
        super(ConvMixerBlock, self).__init__()
        self.block = nn.Sequential(
            * [nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(dim, dim, kernel_size=(7,7), groups=dim, padding=(7//2, 7//2)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )

    def forward(self, x):
        x = self.block(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x




class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x



class CMU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(CMU_Net, self).__init__()

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        self.ConvMixer = ConvMixerBlock(dim=1024, depth=7)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=512*2, ch_out=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=256*2, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=128*2, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=64*2, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.ipalc4 = MSAG(512)
        self.ipalc3 = MSAG(256)
        self.ipalc2 = MSAG(128)
        self.ipalc1 = MSAG(64)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y


    def forward(self, x):
        # encoding path
        # layer1 -> ->
        x1 = self.Conv1(x)  # 64
        # layer2 -> ->
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2) # 128
        # layer3 -> ->
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3) # 256
        # layer4 -> ->
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4) # 512
        # layer5 -> ->
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5) # 1024
        x5 = self.ConvMixer(x5)
    
        x4 = self.ipalc4(x4)
        x3 = self.ipalc3(x3)
        x2 = self.ipalc2(x2)
        x1 = self.ipalc1(x1)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2) 
        d1 = self.Conv_1x1(d2)

        return d1
