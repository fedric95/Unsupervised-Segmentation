#from __future__ import print_function
import torch
import torch.nn.init
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from useg.kanezaki.dfc.model import Net, CustomLoss
from useg.kanezaki.dfc.data import Dataset

scribble = False
nChannel = 100
maxIter = 1000
minLabels = 3
lr = 0.1
nConv = 2
visualize = 1
stepsize_sim = 1.0 # 'step size for similarity loss'
stepsize_con = 1.0 # 'step size for continuity loss'
stepsize_scr = 0.5 # 'step size for scribble loss'
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


scribble_file = None
if scribble:
    scribble_file = input.replace('.'+input.split('.')[-1],'_scribble.png')

input_files = ['C:/Users/federico/Documents/CL/Image15_40.tif']

scribble_files = None

dataset = Dataset(input_files, scribble_files)

in_channels = 1
custom_loss = CustomLoss(stepsize_sim = stepsize_sim, stepsize_con = stepsize_con, stepsize_scr = stepsize_scr, scribble=scribble)
model = Net( 
    in_channels = in_channels, 
    out_channels = nChannel, 
    n_layers = nConv, 
    loss=custom_loss,
    learning_rate=lr
)

trainer = pl.Trainer(
    accelerator=device,
    devices=1,
    max_epochs=maxIter, log_every_n_steps=1
)
trainer.fit(model, DataLoader(dataset, batch_size = None))