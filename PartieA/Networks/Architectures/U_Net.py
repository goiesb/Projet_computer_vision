import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################################
#
# CLASS DESCRIBING A LIGHT U-NET ARCHITECTURE
# 
######################################################################################

class U_Net(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.nb_channel = param["MODEL"]["NB_CHANNEL"]  # base number of channels

        # ---------- Encoder ----------
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, self.nb_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel, self.nb_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(self.nb_channel, self.nb_channel * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel * 2, self.nb_channel * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 2),
            nn.ReLU(inplace=True)
        )

        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)

        # ---------- Bottleneck ----------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.nb_channel * 2, self.nb_channel * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel * 4, self.nb_channel * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 4),
            nn.ReLU(inplace=True)
        )

        # ---------- Decoder ----------
        self.upconv2 = nn.ConvTranspose2d(self.nb_channel * 4, self.nb_channel * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(self.nb_channel * 4, self.nb_channel * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel * 2, self.nb_channel * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 2),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(self.nb_channel * 2, self.nb_channel, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(self.nb_channel * 2, self.nb_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel, self.nb_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True)
        )

        # ---------- Output layer ----------
        self.out_conv = nn.Conv2d(self.nb_channel, 5, kernel_size=1)  # 5 land-use classes

    def forward(self, x):
        # ----- Encoder -----
        x1 = self.enc1(x) # Dimensions: (batch_size, nb_channel, H, W)
        x2 = self.pool(x1) # Dimensions: (batch_size, nb_channel, H/2, W/2)
        x2 = self.enc2(x2) # Dimensions: (batch_size, nb_channel*2, H/2, W/2)

        # ----- Bottleneck -----
        x3 = self.pool(x2) # Dimensions: (batch_size, nb_channel*2, H/4, W/4)
        x3 = self.bottleneck(x3) # Dimensions: (batch_size, nb_channel*4, H/4, W/4)
        # ----- Decoder -----
        x = self.upconv2(x3) # Dimensions: (batch_size, nb_channel*2, H/2, W/2)
        #skip connection
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x) # Dimensions: (batch_size, nb_channel*2, H/2, W/2)

        x = self.upconv1(x) # Dimensions: (batch_size, nb_channel, H, W)
        x = torch.cat([x, x1], dim=1) # Dimensions: (batch_size, nb_channel*2, H, W)
        x = self.dec1(x) # Dimensions: (batch_size, nb_channel, H, W)

        # ----- Output -----
        x = self.out_conv(x) # Dimensions: (batch_size, 5, H, W)
        return x
