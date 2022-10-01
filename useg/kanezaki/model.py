import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# CNN model
class KanezakiNet(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers
    ):
        super(KanezakiNet, self).__init__()
        self.n_layers = n_layers
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(self.n_layers-1):
            self.conv2.append( nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(out_channels) )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.n_layers-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return(x)