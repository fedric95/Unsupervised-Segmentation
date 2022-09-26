
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pytorch_lightning as pl
import cv2
import numpy as np
from useg.kanezaki.model import KanezakiNet

# CNN model
class Net(KanezakiNet):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers,
        loss,
        learning_rate=2e-4
    ):
        super(Net, self).__init__(in_channels, out_channels, n_layers)
        self.learning_rate=learning_rate
        self.loss = loss
        self.label_colours = np.random.randint(255,size=(100,3))
        
    
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



class CustomLoss(nn.Module):
    
    def __init__(
        self, 
        stepsize_sim,
        stepsize_con,
        stepsize_scr = None,
        scribble = False
    ):
        super(CustomLoss, self).__init__()
        self.stepsize_sim = stepsize_sim
        self.stepsize_con = stepsize_con
        self.scribble = scribble

        # continuity loss definition
        self.loss_hpy = nn.L1Loss(size_average = True)
        self.loss_hpz = nn.L1Loss(size_average = True)

        # similarity loss definition
        self.loss_fn = nn.CrossEntropyLoss()

        # scribble loss definition
        if(self.scribble):
            self.loss_fn_scr = nn.CrossEntropyLoss()
            self.stepsize_scr = stepsize_scr
 
    def forward(
        self,
        HPy,
        HPz,
        output, 
        target,
        inds_sim = None,
        inds_scr = None,
        target_scr = None
    ):
        HPy_target = torch.zeros_like(HPy, requires_grad=False)
        HPz_target = torch.zeros_like(HPz, requires_grad=False)

        if self.scribble:
            l = self.stepsize_sim * self.loss_fn(output[ inds_sim ], target[ inds_sim ])
            l = l + self.stepsize_scr * self.loss_fn_scr(output[ inds_scr ], target_scr[ inds_scr ])
        else:
            l = self.stepsize_sim * self.loss_fn(output, target)
        
        l = l + self.stepsize_con * (self.loss_hpy(HPy,HPy_target) + self.loss_hpz(HPz,HPz_target))

        return(l)