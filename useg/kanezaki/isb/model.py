
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pytorch_lightning as pl
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
        self.save_hyperparameters()

        self.learning_rate=learning_rate
        self.label_colours = np.random.randint(255,size=(out_channels,3))
        self.loss = CustomLoss()
    

    def configure_optimizers(self):
        opt = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return(opt)

    def _shared_step(self, batch, batch_idx, step):

        image = batch['image']
        preseg = batch['preseg'] #(NxHxW)

        output = self(image) #(NxCxHxW)
        _, target = torch.max(output, 1, keepdim=True)  #(NxCxHxW)->(Nx1xHxW)

        loss, target_refs, unique_labels = self.loss(output, target, preseg)

        self.log(step+'/loss', loss, prog_bar = True, on_epoch = True)
        self.log(step+'/n_labels', len(np.unique(unique_labels)), prog_bar=True)

        im_target = target[0, 0, :, :].cpu().detach().numpy()
        im_target_ref = target_refs[0, 0, :, :]
        im_preseg = preseg[0, 0, :, :].cpu().detach().numpy()
        if('label' in batch.keys()):
            label = batch['label'][0, 0, :, :].cpu().detach().numpy()
            ari = adjusted_rand_score(label.ravel(), im_target.ravel())
            self.log(step+'/ari', ari, prog_bar=True, on_epoch = True)
            logimage(self.logger, step+'/label', label, self.global_step, self.label_colours)
        
        im_input = image[0, :, :, :].cpu().detach().numpy().transpose([1,2,0])
        for c in range(im_input.shape[-1]):
            im_input[:, :, c] = (im_input[:, :, c]-im_input[:, :, c].min())/(im_input[:, :, c].max()-im_input[:, :, c].min())

        logimage(self.logger, step+'/input',       im_input,       self.global_step)
        logimage(self.logger, step+'/output',      im_target,      self.global_step, self.label_colours)
        logimage(self.logger, step+'/output_ref',  im_target_ref,  self.global_step, self.label_colours)
        logimage(self.logger, step+'/preseg',      im_preseg,      self.global_step, self.label_colours)

        return(loss)

    def training_step(self, batch, batch_idx):
        return(self._shared_step(batch, batch_idx, 'train'))
    
    def validation_step(self, batch, batch_idx):
        return(self._shared_step(batch, batch_idx, 'val'))






class CustomLoss(nn.Module):
    
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')
 

    # superpixel refinement
    # TODO: use Torch Variable instead of numpy for faster calculation
    def superpixel_refinement(self, im_target, l_inds):
        im_target = im_target.cpu().detach().numpy().copy()
        for i in range(len(l_inds)):
            l_ind = l_inds[i]
            labels_per_sp = im_target[l_ind]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[j] )[ 0 ] )
            im_target[l_ind] = u_labels_per_sp[ np.argmax( hist ) ]
        return(im_target)

    def forward(self, output, target, preseg):
        original_shape = [preseg.shape[2], preseg.shape[3]]

        target = target.reshape(output.shape[0], -1) #(Nx(H*W))
        preseg = preseg.reshape(output.shape[0], -1) #(Nx(H*W))
        output = output.reshape(output.shape[0], output.shape[1], -1) #(NxCx(H*W))

        target_refs = []
        unique_labels = []
        loss = 0
        for i in range(preseg.shape[0]):
            u_labels = torch.unique(preseg[i])
            l_inds = []
            for j in range(len(u_labels)):
                ul = torch.where(preseg[i]==u_labels[j])[0]
                l_inds.append(ul.cpu().numpy())
            target_ref = self.superpixel_refinement(target[i], l_inds)
            loss = loss + self.loss(output[i].unsqueeze(0), torch.from_numpy(target_ref).to(output.device).unsqueeze(0))
            unique_labels.extend(np.unique(target_ref).tolist())
            target_ref = np.expand_dims(target_ref, 0)
            target_refs.append(target_ref)
        target_refs = np.concatenate(target_refs, axis=0)
        target_refs = target_refs.reshape([target_refs.shape[0], 1, original_shape[0], original_shape[1]])

        loss = loss/preseg.shape[0]
        return(loss, target_refs, unique_labels)

