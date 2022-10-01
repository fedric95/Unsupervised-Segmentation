import torch
import torch.nn.init
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from useg.kanezaki.isb.model import Net
from useg.kanezaki.isb.data import Dataset

import skimage.segmentation
import skimage.io
import numpy as np
import os.path

n_clusters = 100
epochs = 1000
learning_rate = 0.1
n_layers = 2
visualize = 1
use_gpu = 0
transform = lambda x: x/255.0

if bool(use_gpu)==True:
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print('No GPUs available. Placing the model on CPU..')
        device = 'cpu'
    else:
        print('Placing the model on GPU..')
        device = 'gpu'
else:
    print('Placing the model on CPU..')
    device = 'cpu'

input_dir = r'C:/Users/federico/Documents/CL/'
input_files = os.listdir(input_dir)
input_files = [os.path.join(input_dir,file) for file in input_files if file.endswith('.png') or file.endswith('.jpg') or file.endswith('tif')]

label_files = []
segmentation_directory = 'C:/Users/federico/Documents/SEG/'
segmentation_args = {'compactness': 10, 'n_segments': 100000, 'start_label':0, 'multichannel': False}
for i in range(len(input_files)):
    labels = skimage.segmentation.slic(skimage.io.imread(input_files[i]), **segmentation_args)
    
    labels_path = os.path.join(segmentation_directory, 'image_'+str(i)+'.tif')
    skimage.io.imsave(labels_path, labels)
    assert np.prod(labels==skimage.io.imread(labels_path))==1, 'I/O error'

    label_files.append(labels_path)

dataset = Dataset(input_files, label_files=label_files, transform=transform)

# train
in_channels = 1
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