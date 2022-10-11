import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch.optim as optim
from useg.wnet.loss import SoftCutLoss
import pytorch_lightning as pl
from useg.utils import logimage

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, seperable=True):
        super(Block, self).__init__()
        
        if seperable:
            
            self.spatial1=nn.Conv2d(in_filters, in_filters, kernel_size=3, groups=in_filters, padding=1)
            self.depth1=nn.Conv2d(in_filters, out_filters, kernel_size=1)
            
            self.conv1=lambda x: self.depth1(self.spatial1(x))
            
            self.spatial2=nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, groups=out_filters)
            self.depth2=nn.Conv2d(out_filters, out_filters, kernel_size=1)
            
            self.conv2=lambda x: self.depth2(self.spatial2(x))
            
        else:
            
            self.conv1=nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
            self.conv2=nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1)
        
        
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.65) 
        self.batchnorm1=nn.BatchNorm2d(out_filters)
        
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.65) 
        self.batchnorm2=nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.batchnorm1(self.conv1(x)).clamp(0)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.batchnorm2(self.conv2(x)).clamp(0)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, hiddden_dim=64, out_channels=3):
        super(UNet, self).__init__()
        
        self.enc1=Block(in_channels, hiddden_dim, seperable=False)
        self.enc2=Block(hiddden_dim, 2*hiddden_dim)
        self.enc3=Block(2*hiddden_dim, 4*hiddden_dim)
        self.enc4=Block(4*hiddden_dim, 8*hiddden_dim)
        
        self.middle=Block(8*hiddden_dim, 16*hiddden_dim)
        
        self.up1=nn.ConvTranspose2d(16*hiddden_dim, 8*hiddden_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1=Block(16*hiddden_dim, 8*hiddden_dim)
        self.up2=nn.ConvTranspose2d(8*hiddden_dim, 4*hiddden_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2=Block(8*hiddden_dim, 4*hiddden_dim)
        self.up3=nn.ConvTranspose2d(4*hiddden_dim, 2*hiddden_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3=Block(4*hiddden_dim, 2*hiddden_dim)
        self.up4=nn.ConvTranspose2d(2*hiddden_dim, hiddden_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4=Block(2*hiddden_dim, hiddden_dim, seperable=False)
        
        self.final=nn.Conv2d(hiddden_dim, out_channels, kernel_size=(1, 1))

        
    def forward(self, x):
        
        enc1=self.enc1(x)
        
        enc2=self.enc2(F.max_pool2d(enc1, (2, 2)))
        
        enc3=self.enc3(F.max_pool2d(enc2, (2,2)))
        
        enc4=self.enc4(F.max_pool2d(enc3, (2,2)))
        
        
        middle=self.middle(F.max_pool2d(enc4, (2,2)))
        
        
        up1=torch.cat([enc4, self.up1(middle)], 1)
        dec1=self.dec1(up1)
        
        up2=torch.cat([enc3, self.up2(dec1)], 1)
        dec2=self.dec2(up2)
        
        up3=torch.cat([enc2, self.up3(dec2)], 1)
        dec3=self.dec3(up3)
        
        up4=torch.cat([enc1, self.up4(dec3)], 1)
        dec4=self.dec4(up4)
        
        
        final=self.final(dec4)
        
        return final

class WNet(pl.LightningModule):
    def __init__(
        self, 
        in_channels=3, 
        hiddden_dim=64, 
        intermediate_channels=4,
        learning_rate_clust = 2e-4,
        learning_rate_recon = 2e-4,
        radius = 5
    ):
        super(WNet, self).__init__()
        self.save_hyperparameters()


        self.automatic_optimization = False
        
        
        self.encoder=UNet(in_channels=in_channels, hiddden_dim = hiddden_dim, out_channels = intermediate_channels)
        self.decoder=UNet(in_channels=intermediate_channels, hiddden_dim = hiddden_dim, out_channels = in_channels)

        self.label_colours = np.random.randint(255,size=(100,3))
        self.intermediate_channels = intermediate_channels
        self.learning_rate_clust=learning_rate_clust        
        self.learning_rate_recon=learning_rate_recon      

        self.loss_clust = SoftCutLoss(radius=radius)
        self.loss_recon = torch.nn.MSELoss()

        self.label_colours = np.random.randint(255,size=(intermediate_channels,3))

    def forward(self, x, returns='both'):

        enc = self.encoder(x)

        if returns=='enc':
            return enc
        
        dec=self.decoder(F.softmax(enc, 1))
        
        if returns=='dec':
            return dec
        
        if returns=='both':
            return enc, dec
        
        else:
            raise ValueError('Invalid returns, returns must be in [enc dec both]')


    def configure_optimizers(self):
        optim_clust = optim.SGD(self.parameters(), lr=self.learning_rate_clust, momentum=0.9)
        optim_recon = optim.SGD(self.parameters(), lr=self.learning_rate_recon, momentum=0.9)
        return optim_clust, optim_recon

    def _shared_step(self, batch, batch_idx, step):
        image = batch['image']
        optim_clust, optim_recon = self.optimizers()
        optim_clust.zero_grad()
        enc = self(image, returns='enc')
        
        n_cut_loss=self.loss_clust(image,  F.softmax(enc, 1))

        self.manual_backward(n_cut_loss)
        optim_clust.step()
        optim_clust.zero_grad()
        dec = self(image, returns='dec')
        rec_loss=self.loss_recon(image, dec)
        self.manual_backward(rec_loss)
        optim_recon.step()


        im_input  = image[0].cpu().detach().numpy().transpose([1,2,0])
        im_output = dec[0].cpu().detach().numpy().transpose([1,2,0])

        for c in range(im_input.shape[-1]):
            im_input[:, :, c] = (im_input[:, :, c]-im_input[:, :, c].min())/(im_input[:, :, c].max()-im_input[:, :, c].min())

        logimage(self.logger, step+'/input',  im_input,  self.global_step)
        logimage(self.logger, step+'/output', im_output, self.global_step)

        self.log(step+'/reconstruction_loss', rec_loss.detach().item(), prog_bar=True)
        self.log(step+'/soft_n_cut_loss', n_cut_loss.detach().item(), prog_bar=True)



        #visualize = True
        #if visualize:
        #    ignore, target = torch.max(F.softmax(enc.detach(), 1)[0, :, :, :], 0)
        #    im_target = target.data.cpu().numpy()
        #    im_target_rgb = np.array([self.label_colours[ c % self.intermediate_channels ] for c in im_target])
        #    im_target_rgb = im_target_rgb.reshape( [x.shape[-2], x.shape[-1], 3] ).astype( np.uint8 )
        #    cv2.imwrite( "output.png", im_target_rgb )
    
    def training_step(self, batch, batch_idx):
        return(self._shared_step(batch, batch_idx, 'train'))

    def validation_step(self, batch, batch_idx):
        return(self._shared_step(batch, batch_idx, 'val'))