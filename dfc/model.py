
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pytorch_lightning as pl
import cv2
import numpy as np


# CNN model
class Net(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers,
        loss,
        learning_rate=2e-4
    ):
        super(Net, self).__init__()
        self.n_layers = n_layers
        self.out_channles = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(self.n_layers-1):
            self.conv2.append( nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(out_channels) )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(out_channels)

        
        self.learning_rate=learning_rate
        self.loss = loss
        self.label_colours = np.random.randint(255,size=(100,3))
        
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
    

    def configure_optimizers(self):
        opt = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return(opt)


    

    def training_step(self, batch, batch_idx):

        image = batch['image']
        inds_sim   = batch.get('inds_sim', None)
        inds_scr   = batch.get('inds_scr', None)
        target_scr = batch.get('target_scr', None)
        
        output = self(image)
        output = output[ 0 ].permute( 1, 2, 0 ).contiguous().view( -1, self.out_channles )
        outputHP = output.reshape( (image.shape[-2], image.shape[-1], self.out_channles) )

        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]

        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        loss = self.loss(HPy = HPy, HPz = HPz, output = output, target = target, inds_sim = inds_sim, inds_scr = inds_scr, target_scr = target_scr)

        visualize = True
        if visualize:
            im_target_rgb = np.array([self.label_colours[ c % self.out_channles ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( [image.shape[-2], image.shape[-1], 3] ).astype( np.uint8 )
            cv2.imwrite( "output.png", im_target_rgb )

        self.log('nLabels', nLabels, prog_bar=True)

        return(loss)
