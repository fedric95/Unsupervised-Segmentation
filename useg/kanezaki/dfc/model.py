
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pytorch_lightning as pl
import cv2
import numpy as np
from useg.kanezaki.model import KanezakiNet

# CNN model
# batch_size != None is not supported
class Net(KanezakiNet):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers,
        stepsize_sim, 
        stepsize_con,
        learning_rate=2e-4
    ):
        super(Net, self).__init__(in_channels, out_channels, n_layers)
        self.learning_rate=learning_rate
        self.loss = CustomLoss(stepsize_sim = stepsize_sim, stepsize_con = stepsize_con)
        self.label_colours = np.random.randint(255,size=(100,3))
        
    
    def configure_optimizers(self):
        opt = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return(opt)


    def training_step(self, batch, batch_idx): #batch_size == 1

        image = batch['image']

        output = self(image)
        output = output.permute(0, 2, 3, 1) #(NxCxHxW)->(NxHxWxC)
        _, target = torch.max(output.reshape(output.shape[0], output.shape[1]*output.shape[2], output.shape[3]),2)

        unique_labels = []
        loss = 0
        for i in range(image.shape[0]):
            loss = loss + self.loss(output = output[i], target = target[i]) 
            unique_labels.extend(torch.unique(target[i]).cpu().numpy().tolist())

        loss = loss/image.shape[0]
        nLabels = len(np.unique(unique_labels))
        
        visualize = True
        if visualize:
            im_target = target[i].cpu().numpy()
            im_target_rgb = np.array([self.label_colours[ c % self.out_channels ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( [image.shape[-2], image.shape[-1], 3] ).astype( np.uint8 )
            cv2.imwrite( "output.png", im_target_rgb )

        self.log('nLabels', nLabels, prog_bar=True)

        return(loss)



class CustomLoss(nn.Module):
    
    def __init__(self, stepsize_sim, stepsize_con):
        super(CustomLoss, self).__init__()
        self.stepsize_sim = stepsize_sim
        self.stepsize_con = stepsize_con

        # similarity loss definition
        self.loss_fn = nn.CrossEntropyLoss()
 
    def forward(self, output, target):
        
        HPy = output[1:, :, :] - output[0:-1, :, :]
        HPz = output[:, 1:, :] - output[:, 0:-1, :]
        
        
        l = self.stepsize_sim * self.loss_fn(output.reshape((output.shape[0]*output.shape[1], output.shape[2])), target)
        l = l + self.stepsize_con * (torch.norm(HPy, p=1)/np.prod(HPy.shape) + torch.norm(HPz,p=1)/np.prod(HPz.shape))

        return(l)