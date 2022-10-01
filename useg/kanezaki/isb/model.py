
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
        learning_rate=2e-4
    ):
        super(Net, self).__init__(in_channels, out_channels, n_layers)
        self.learning_rate=learning_rate
        self.label_colours = np.random.randint(255,size=(100,3))
        self.loss = nn.CrossEntropyLoss(reduction='mean')
    

    def configure_optimizers(self):
        opt = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return(opt)


    def training_step(self, batch, batch_idx): #implemented for single batch
        image = batch['image']
        labels = batch['labels']

        labels = labels.reshape(labels.shape[0], labels.shape[1]*labels.shape[2]).cpu()

        output = self(image)
        output = output.permute(0, 2, 3, 1) #(NxCxHxW)->(NxHxWxC)
        output = output.reshape(output.shape[0], output.shape[1]*output.shape[2], output.shape[3])
        _, target = torch.max(output,2)
        
        

        unique_labels = []
        loss = 0
        for i in range(labels.shape[0]):
            u_labels = torch.unique(labels[i])
            l_inds = []
            for j in range(len(u_labels)):
                ul = torch.where(labels[i]==u_labels[j])[0]
                l_inds.append(ul.numpy())
            
            im_target = self.superpixel_refinement(target[i], l_inds)
            loss += self.loss(output[i], torch.from_numpy(im_target).to(self.device))

            unique_labels.extend(np.unique(im_target).tolist())
        loss = loss/labels.shape[0]

        
        nLabels = len(np.unique(unique_labels))
        visualize = True
        if visualize:
            im_target_rgb = np.array([self.label_colours[ c % self.out_channels ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( [image.shape[-2], image.shape[-1], 3] ).astype( np.uint8 )
            cv2.imwrite( "output.png", im_target_rgb )

        self.log('nLabels', nLabels, prog_bar=True)

        return(loss)
    

    # superpixel refinement
    # TODO: use Torch Variable instead of numpy for faster calculation
    def superpixel_refinement(self, im_target, l_inds):
        im_target = im_target.cpu().numpy().copy()
        for i in range(len(l_inds)):
            l_ind = l_inds[i]
            labels_per_sp = im_target[l_ind]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[j] )[ 0 ] )
            im_target[l_ind] = u_labels_per_sp[ np.argmax( hist ) ]
        return(im_target)