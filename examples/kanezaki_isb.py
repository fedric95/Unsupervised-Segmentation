import torch
import torch.nn.init
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from useg.kanezaki.isb.model import Net
from useg.kanezaki.isb.data import Dataset
from useg.utils import isimage, getdevice

import skimage.segmentation
import skimage.io
import numpy as np
import os.path

n_clusters = 100
epochs = 1000
learning_rate = 0.1
n_layers = 2
in_channels = 1
use_gpu = 0

def transform(out):
    out['image'] = out['image']/255.0
    return(out)

device = getdevice(use_gpu)

input_dir = r'C:/Users/federico/Documents/CL/'
input_files = os.listdir(input_dir)
input_files = [os.path.join(input_dir,file) for file in input_files if isimage(file)]

preseg_files = []
segmentation_directory = 'C:/Users/federico/Documents/SEG/'
segmentation_args = {'compactness': 10, 'n_segments': 100000, 'start_label':0, 'multichannel': False}
for i in range(len(input_files)):
    preseg = skimage.segmentation.slic(skimage.io.imread(input_files[i]), **segmentation_args)
    preseg_path = os.path.join(segmentation_directory, 'image_'+str(i)+'.tif')
    skimage.io.imsave(preseg_path, preseg)
    assert np.prod(preseg==skimage.io.imread(preseg_path))==1, 'I/O error'

    preseg_files.append(preseg_path)

dataset = Dataset(input_files, preseg_files=preseg_files, transform=transform)

model = Net( 
    in_channels = in_channels, 
    out_channels = n_clusters, 
    n_layers = n_layers, 
    learning_rate=learning_rate
)

trainer = pl.Trainer(
    accelerator=device,
    devices=1,
    max_epochs=epochs, log_every_n_steps=1
)
trainer.fit(model, DataLoader(dataset, batch_size = 2))