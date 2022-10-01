#from __future__ import print_function
import torch
import torch.nn.init
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from useg.kanezaki.dfc.model import Net, CustomLoss
from useg.kanezaki.dfc.data import Dataset
import os



n_clusters = 80
epochs = 1000
learning_rate = 0.1
n_layers = 2
visualize = 1
stepsize_sim = 1.0 # 'step size for similarity loss'
stepsize_con = 1.0 # 'step size for continuity loss'
use_gpu = 1
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

dataset = Dataset(input_files, transform=transform)

in_channels = 1
model = Net( 
    in_channels = in_channels, 
    out_channels = n_clusters, 
    n_layers = n_layers, 
    stepsize_sim=stepsize_sim,
    stepsize_con=stepsize_con,
    learning_rate=learning_rate
)

trainer = pl.Trainer(
    accelerator=device,
    devices=1,
    max_epochs=epochs, 
    log_every_n_steps=1
)
trainer.fit(model, DataLoader(dataset, batch_size = 1))