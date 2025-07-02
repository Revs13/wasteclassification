import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class WasteClassifierLite(nn.Module):
    def __init__(self):
        super(WasteClassifierLite, self).__init__()

        #the input here is of shape [batch size, 3, 64, 64] because
        #picture means RGB values means 3 features, and it's 64 x 64
        #out_channels=8 means the computer will come up with 8 features for the images

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #we flattened so that the output of conv_layer() is [batch size, 8, 32, 32]
        #so 8 * 32 * 32 = 8192 features

        self.fc_layer = nn.Sequential(nn.Linear(8192, 100), nn.ReLU(), 
                                      nn.Linear(100, 1), nn.Sigmoid())
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WasteClassifierLite().to(device)
