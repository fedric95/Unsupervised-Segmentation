import torch
import torch.nn.init
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from useg.kanezaki.isb.model import Net
from useg.kanezaki.isb.data import Dataset


nChannel = 100
maxIter = 1000
minLabels = 3
lr = 0.1
nConv = 2
num_superpixels = 10000
compactness = 100
visualize = 1
use_gpu = 0


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


segmentation_args = {'compactness': compactness, 'n_segments': num_superpixels, 'start_label':0}

input_files = [
    'C:/Users/federico/Documents/CL/Image15_40.tif', 
    'C:/Users/federico/Documents/CL/Image16_40.tif', 
    'C:/Users/federico/Documents/CL/Image17_40.tif',
    'C:/Users/federico/Documents/CL/Image18_40.tif',
    'C:/Users/federico/Documents/CL/Image19_40.tif',
    'C:/Users/federico/Documents/CL/Image20_40.tif',
    'C:/Users/federico/Documents/CL/Image21_40.tif'
]
dataset = Dataset(input_files, segmentation_args=segmentation_args)

# train
in_channels = 1
model = Net( 
    in_channels = in_channels, 
    out_channels = nChannel, 
    n_layers = nConv, 
    learning_rate=lr
)

trainer = pl.Trainer(
    accelerator=device,
    devices=1,
    max_epochs=maxIter, log_every_n_steps=1
)
trainer.fit(model, DataLoader(dataset, batch_size = None))