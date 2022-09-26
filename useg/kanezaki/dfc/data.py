import cv2
import torch
import numpy as np


class Dataset:

    def __init__(
        self, 
        input_files,
        scrible_files = None
    ):
        if(scrible_files is not None):
            assert len(input_files) == len(scrible_files), 'Len of scribble files must be equal to the len of input files'
        
        self.input_files = input_files
        self.scrible_files = scrible_files

    def __len__(self):
        return(len(self.input_files))

    def __getitem__(self, idx):


        out = {}
        im = cv2.imread(self.input_files[idx]).transpose( (2, 0, 1) )[:1, :, :]
        out['image'] = torch.tensor( np.array([im.astype('float32')/255.]))
        
        if(self.scrible_files is not None):
            mask = cv2.imread(self.scrible_files[idx],-1)
            mask = mask.reshape(-1)
            mask_inds = np.unique(mask)
            mask_inds = np.delete( mask_inds, np.argwhere(mask_inds==255) )

            out['inds_sim'] = torch.tensor(np.where( mask == 255 )[ 0 ])
            out['inds_scr'] = torch.tensor( np.where( mask != 255 )[ 0 ] )
            out['target_scr'] = torch.tensor( mask.astype(np.int) )
            out['classes'] = len(mask_inds)
        
        return(out)
