import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################################
#
# CLASS DESCRIBING AN ATTENTION U-NET ARCHITECTURE
#
######################################################################################

class AttentionBlock(nn.Module):
    """
    Attention gate for skip connections in Attention U-Net
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # element-wise multiplication


class ConvBlock(nn.Module):
    """
    Double convolution block with BatchNorm and LeakyReLU
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Attention_U_Net(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.nb_channel = param["MODEL"]["NB_CHANNEL"]  # base channel size

        # Encoder
        self.conv1 = ConvBlock(3, self.nb_channel)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(self.nb_channel, self.nb_channel * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(self.nb_channel * 2, self.nb_channel * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(self.nb_channel * 4, self.nb_channel * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(self.nb_channel * 8, self.nb_channel * 4, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=self.nb_channel * 4, F_l=self.nb_channel * 4, F_int=self.nb_channel * 2)
        self.dec3 = ConvBlock(self.nb_channel * 8, self.nb_channel * 4)

        self.up2 = nn.ConvTranspose2d(self.nb_channel * 4, self.nb_channel * 2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=self.nb_channel * 2, F_l=self.nb_channel * 2, F_int=self.nb_channel)
        self.dec2 = ConvBlock(self.nb_channel * 4, self.nb_channel * 2)

        self.up1 = nn.ConvTranspose2d(self.nb_channel * 2, self.nb_channel, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=self.nb_channel, F_l=self.nb_channel, F_int=self.nb_channel // 2)
        self.dec1 = ConvBlock(self.nb_channel * 2, self.nb_channel)

        # Output layer
        self.final = nn.Conv2d(self.nb_channel, 5, kernel_size=1)  # 5 classes

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        p1 = self.pool1(x1)

        x2 = self.conv2(p1)
        p2 = self.pool2(x2)

        x3 = self.conv3(p2)
        p3 = self.pool3(x3)

        x4 = self.conv4(p3)

        # Decoder
        d3 = self.up3(x4)
        x3 = self.att3(g=d3, x=x3)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=x2)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=x1)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out
