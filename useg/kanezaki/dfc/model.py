
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pytorch_lightning as pl
import cv2
import numpy as np
from useg.kanezaki.model import KanezakiNet
from useg.utils import logimage
from sklearn.metrics.cluster import adjusted_rand_score
from torchvision.utils import make_grid

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
        self.save_hyperparameters()

        self.learning_rate=learning_rate
        self.loss = CustomLoss(stepsize_sim = stepsize_sim, stepsize_con = stepsize_con)
        self.label_colours = np.random.randint(255,size=(out_channels,3))
        
    
    def configure_optimizers(self):
        opt = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return(opt)

    def _shared_step(self, batch, batch_idx, step):

        image = batch['image']

        output = self(image)
        _, target = torch.max(output, 1, keepdim=True)  #(NxCxHxW)->(Nx1xHxW)
        loss = self.loss(output = output, target = target)
        
        self.log(step+'/loss', loss, prog_bar = True, on_epoch = True)
        self.log(step+'/n_labels', len(np.unique(np.unique(target.cpu().numpy()))), prog_bar=True)
        

        im_target = target[0, 0, :, :].cpu().detach().numpy()
        if('label' in batch.keys()):
            label = batch['label'][0, 0, :, :].cpu().detach().numpy()
            ari = adjusted_rand_score(label.ravel(), im_target.ravel())
            self.log(step+'/ari', ari, prog_bar=True, on_epoch = True)
            logimage(self.logger, step+'/label', label, self.global_step, self.label_colours)
        
        im_input = image[0, :, :, :].cpu().detach().numpy().transpose([1,2,0])
        for c in range(im_input.shape[-1]):
            im_input[:, :, c] = (im_input[:, :, c]-im_input[:, :, c].min())/(im_input[:, :, c].max()-im_input[:, :, c].min())

        logimage(self.logger, step+'/output', im_target, self.global_step, self.label_colours)
        logimage(self.logger, step+'/input', im_input, self.global_step)

        
        return(loss)

    def training_step(self, batch, batch_idx):
        return(self._shared_step(batch, batch_idx, 'train'))

    def validation_step(self, batch, batch_idx):
        return(self._shared_step(batch, batch_idx, 'val'))

class CustomLoss(nn.Module):
    
    def __init__(self, stepsize_sim, stepsize_con):
        super(CustomLoss, self).__init__()
        self.stepsize_sim = stepsize_sim
        self.stepsize_con = stepsize_con

        # similarity loss definition
        self.loss_fn = nn.CrossEntropyLoss()
 
    def forward(self, output, target):

        target = target.reshape(target.shape[0], -1)
        output = output.permute(0, 2, 3, 1) #(NxCxHxW)->(NxHxWxC)

        loss = 0
        for i in range(output.shape[0]):

            HPy = output[i][1:, :, :] - output[i][0:-1, :, :]
            HPz = output[i][:, 1:, :] - output[i][:, 0:-1, :]
            
            loss_1 = self.stepsize_sim * self.loss_fn(output[i].reshape((output[i].shape[0]*output[i].shape[1], output[i].shape[2])), target[i])
            loss_2 = self.stepsize_con * (torch.norm(HPy, p=1)/np.prod(HPy.shape) + torch.norm(HPz,p=1)/np.prod(HPz.shape))

            loss = loss + (loss_1+loss_2)

        loss = loss/output.shape[0]

        return(loss)