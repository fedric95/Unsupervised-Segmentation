import torch
import numpy as np
import cv2
from skimage import segmentation

class Dataset:

    def __init__(
        self, 
        input_files,
        segmentation_args,
    ):
        self.input_files = input_files
        self.segmentation_args = segmentation_args

    def __len__(self):
        return(len(self.input_files))

    def __getitem__(self, idx):


        out = {}
        im = cv2.imread(self.input_files[idx]).transpose( (2, 0, 1) )[:1, :, :]
        out['image'] = torch.tensor( np.array([im.astype('float32')/255.]))
        
        # slic
        
        labels = segmentation.slic(im, **self.segmentation_args) #see how to set seed
        labels = labels.reshape(im.shape[0]*im.shape[1])
        u_labels = np.unique(labels)
        out['l_inds'] = []
        for i in range(len(u_labels)):
            out['l_inds'].append( np.where( labels == u_labels[ i ] )[ 0 ] )
        
        
        return(out)
