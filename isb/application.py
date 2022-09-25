#from __future__ import print_function
import argparse
import torch
import torch.nn.init
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import Net
from data import Dataset

import dfc.loss

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int, 
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=100, type=float, 
                    help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
                    
parser.add_argument('--use_gpu', required=False, type=int, default=1,
                    help='Set 1 to use the available GPU. 0 Otherwise')
args = parser.parse_args()



if bool(args.use_gpu)==True:
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


segmentation_args = {'compactness': args.compactness, 'n_segments': args.num_superpixels, 'start_label':0}

input_files = []
dataset = Dataset(input_files, segmentation_args=segmentation_args)

# train
in_channels = 1
model = Net( 
    in_channels = in_channels, 
    out_channels = args.nChannel, 
    n_layers = args.nConv, 
    learning_rate=args.lr
)

trainer = pl.Trainer(
    accelerator=device,
    devices=1,
    max_epochs=args.maxIter, log_every_n_steps=1
)
trainer.fit(model, DataLoader(dataset, batch_size = None))