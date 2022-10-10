
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pytorch_lightning as pl
import cv2
import numpy as np
from useg.kanezaki.model import KanezakiNet
from sklearn.metrics.cluster import adjusted_rand_score
from useg.utils import logimage

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

    def _shared_step(self, batch, batch_idx, step):

        image = batch['image']
        preseg = batch['preseg']

        output = self(image)
        output = output.permute(0, 2, 3, 1) #(NxCxHxW)->(NxHxWxC)
        _, target = torch.max(output.reshape(output.shape[0], output.shape[1]*output.shape[2], output.shape[3]),2)

        preseg = preseg.reshape(preseg.shape[0], preseg.shape[2]*preseg.shape[3]).cpu()        

        unique_labels = []
        loss = 0
        for i in range(preseg.shape[0]):
            u_labels = torch.unique(preseg[i])
            l_inds = []
            for j in range(len(u_labels)):
                ul = torch.where(preseg[i]==u_labels[j])[0]
                l_inds.append(ul.numpy())
            im_target = self.superpixel_refinement(target[i], l_inds)

            loss = loss + self.loss(output[i], torch.from_numpy(im_target).to(self.device))
            unique_labels.extend(np.unique(im_target).tolist())
        loss = loss/preseg.shape[0]

        self.log(step+'/loss', loss, prog_bar = True)
        self.log(step+'/n_labels', len(np.unique(unique_labels)), prog_bar=True)


        if('label' in batch.keys()):
            label = batch['label'].reshape(image.shape[0], image.shape[2]*image.shape[3])
            aris = [adjusted_rand_score(torch.flatten(label[i]).cpu().numpy(), torch.flatten(target[i]).cpu().numpy()) for i in range(image.shape[0])]
            im_label = batch['label'][0, 0, :, :].cpu().detach().numpy()
            im_label = np.array([self.label_colours[ c % self.out_channels ] for c in im_label])/255
            
            self.log(step+'/ari', np.mean(aris), prog_bar=True)
            logimage(self.logger, step+'/label', im_label, self.global_step)

        im_target = np.array([self.label_colours[ c % self.out_channels ] for c in target[0].cpu().numpy()])/255
        im_target = im_target.reshape([image.shape[2], image.shape[3], 3])
        im_input = image[0][:, :, :].cpu().numpy().transpose([1,2,0])

        for c in range(im_input.shape[-1]):
            im_input[:, :, c] = (im_input[:, :, c]-im_input[:, :, c].min())/(im_input[:, :, c].max()-im_input[:, :, c].min())

        logimage(self.logger, step+'/output', im_target, self.global_step)
        logimage(self.logger, step+'/input', im_input, self.global_step)

        return(loss)

    def training_step(self, batch, batch_idx):
        return(self._shared_step(batch, batch_idx, 'train'))
    
    def validation_step(self, batch, batch_idx):
        return(self._shared_step(batch, batch_idx, 'val'))

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




class CustomLoss(nn.Module):
    
    def __init__(self):
        pass
 
    def forward(self, output, target):
        pass