import cv2
import torch
import numpy as np
import skimage.io

class Dataset:

    def __init__(self, input_files, transform=None):
        
        self.input_files = input_files
        self.transform = transform

    def __len__(self):
        return(len(self.input_files))

    def __getitem__(self, idx):

        out = {}

        im = skimage.io.imread(self.input_files[idx])
        if(len(im.shape)==2):
            im = np.expand_dims(im, -1)
        assert len(im.shape)==3, 'Error reading the image'

        im = im.transpose((2, 0, 1)).astype(np.float32)
        if(self.transform is not None):
            im = self.transform(im)
        
        out['image'] = torch.tensor(im)
        return(out)
