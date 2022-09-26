
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
        learning_rate=2e-4
    ):
        super(Net, self).__init__(in_channels, out_channels, n_layers)
        self.learning_rate=learning_rate
        self.label_colours = np.random.randint(255,size=(100,3))
        self.loss = nn.CrossEntropyLoss()
    

    def configure_optimizers(self):
        opt = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return(opt)


    def training_step(self, batch, batch_idx):
        image = batch['image']
        l_inds = batch['l_inds']

        output = self(image)
        output = output[0].permute( 1, 2, 0 ).contiguous().view( -1, self.out_channles )

        ignore, target = torch.max( output, 1 )

        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        # superpixel refinement
        # TODO: use Torch Variable instead of numpy for faster calculation
        for i in range(len(l_inds)):
            labels_per_sp = im_target[ l_inds[ i ] ]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
            im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
        target = torch.tensor(im_target, requires_grad=False, device=self.device)

        loss = self.loss(output, target)

        visualize = True
        if visualize:
            im_target_rgb = np.array([self.label_colours[ c % self.out_channles ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( [image.shape[-2], image.shape[-1], 3] ).astype( np.uint8 )
            cv2.imwrite( "output.png", im_target_rgb )

        self.log('nLabels', nLabels, prog_bar=True)

        return(loss)