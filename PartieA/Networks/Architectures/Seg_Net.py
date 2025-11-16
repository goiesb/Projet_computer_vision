import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################################
#
# CLASS DESCRIBING A SIMPLE SEGNET ARCHITECTURE
# 
######################################################################################

class Seg_Net(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.nb_channel = param["MODEL"]["NB_CHANNEL"]  # base number of channels
        self.num_classes = 5  # 5 land-use classes

        # -------------------
        # ENCODER
        # -------------------
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, self.nb_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel, self.nb_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc2 = nn.Sequential(
            nn.Conv2d(self.nb_channel, self.nb_channel*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel*2, self.nb_channel*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel*2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        # -------------------
        # DECODER
        # -------------------
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(self.nb_channel*2, self.nb_channel*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel*2, self.nb_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True)
        )

        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(self.nb_channel, self.nb_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nb_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nb_channel, self.num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # ---------- ENCODER ----------
        x1 = self.enc1(x)
        size1 = x1.size()
        x1, idx1 = self.pool1(x1)

        x2 = self.enc2(x1)
        size2 = x2.size()
        x2, idx2 = self.pool2(x2)

        # ---------- DECODER ----------
        x = self.unpool2(x2, idx2, output_size=size2)
        x = self.dec2(x)

        x = self.unpool1(x, idx1, output_size=size1)
        x = self.dec1(x)

        return x
