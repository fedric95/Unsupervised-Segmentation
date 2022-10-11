import torch
import numpy as np
from skimage import segmentation
import skimage.io
from tqdm import tqdm

class Dataset:

    def __init__(
        self, 
        input_files,
        preseg_files, #change name to preseg_files
        label_files = None, 
        transform=None
    ):  
        assert len(input_files) == len(preseg_files), 'The len of input files and preseg files must be equal'
        
        if(label_files is not None):
            assert len(input_files) == len(label_files), 'The len of input files and label files must be equal'

        self.input_files = input_files
        self.preseg_files = preseg_files
        self.label_files = label_files
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
        
        out['image'] = torch.tensor(im)

        out['preseg'] = skimage.io.imread(self.preseg_files[idx])
        out['preseg']  = np.expand_dims(out['preseg'], 0)
        out['preseg'] = torch.tensor(out['preseg'], dtype=torch.int64)

        if(self.label_files is not None):
            out['label'] = skimage.io.imread(self.label_files[idx])
            out['label'] = np.expand_dims(out['label'], 0)
            out['label'] = torch.tensor(out['label'])
        
        if(self.transform is not None):
            out = self.transform(out)

        return(out)
