import torch
import numpy as np
from skimage import segmentation
import skimage.io
from tqdm import tqdm

class Dataset:

    def __init__(
        self, 
        input_files,
        label_files
    ):  
        
        self.input_files = input_files
        self.label_files = label_files


    def __len__(self):
        return(len(self.input_files))

    def __getitem__(self, idx):

        out = {}

        im = skimage.io.imread(self.input_files[idx])
        if(len(im.shape)==2):
            im = np.expand_dims(im, -1)
        assert len(im.shape)==3, 'Error reading the image'
        
        im = im.transpose((2, 0, 1)).astype(np.float32)/255.0
        out['image'] = torch.tensor(im)
        out['image'] = out['image'].unsqueeze(0)


        labels = skimage.io.imread(self.label_files[idx])
        labels = labels.reshape(labels.shape[0]*labels.shape[1])
        u_labels = np.unique(labels)
        out['l_inds'] = []
        for i in range(len(u_labels)):
            out['l_inds'].append( np.where( labels == u_labels[ i ] )[ 0 ] )

        return(out)
