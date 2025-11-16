import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################################
# CLASS: U_Net with Dropout
# Description:
#   - Based on a light U-Net architecture.
#   - Uses MaxPooling for downsampling.
#   - Adds Dropout layers to reduce overfitting.
######################################################################################

class Deep_U_Net(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.nb_channel = param["MODEL"]["NB_CHANNEL"]  # base number of feature channels

        # ---------- Encoder ----------
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, self.nb_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel, self.nb_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2)  # Dropout added here
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(self.nb_channel, self.nb_channel * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel * 2, self.nb_channel * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 2),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3)  # Slightly higher dropout for deeper layers
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # ---------- Bottleneck ----------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.nb_channel * 2, self.nb_channel * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel * 4, self.nb_channel * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 4),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.4)  # Stronger dropout at the bottleneck
        )

        # ---------- Decoder ----------
        self.upconv2 = nn.ConvTranspose2d(self.nb_channel * 4, self.nb_channel * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(self.nb_channel * 4, self.nb_channel * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel * 2, self.nb_channel * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel * 2),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3)
        )

        self.upconv1 = nn.ConvTranspose2d(self.nb_channel * 2, self.nb_channel, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(self.nb_channel * 2, self.nb_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel, self.nb_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2)
        )

        # ---------- Output layer ----------
        self.out_conv = nn.Conv2d(self.nb_channel, 5, kernel_size=1)  # 5 segmentation classes

    def forward(self, x):
        # ----- Encoder -----
        x1 = self.enc1(x)  # (B, nb_channel, H, W)
        x2 = self.pool(x1)  # (B, nb_channel, H/2, W/2)
        x2 = self.enc2(x2)  # (B, nb_channel*2, H/2, W/2)

        # ----- Bottleneck -----
        x3 = self.pool(x2)  # (B, nb_channel*2, H/4, W/4)
        x3 = self.bottleneck(x3)  # (B, nb_channel*4, H/4, W/4)

        # ----- Decoder -----
        x = self.upconv2(x3)  # (B, nb_channel*2, H/2, W/2)
        x = torch.cat([x, x2], dim=1)  # skip connection
        x = self.dec2(x)  # (B, nb_channel*2, H/2, W/2)

        x = self.upconv1(x)  # (B, nb_channel, H, W)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)  # (B, nb_channel, H, W)

        # ----- Output -----
        x = self.out_conv(x)  # (B, 5, H, W)
        return x
