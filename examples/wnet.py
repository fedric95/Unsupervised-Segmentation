# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:38:02 2018
@author: Tao Lin

Implementation of the W-Net unsupervised image segmentation architecture
"""

import argparse
import datetime

from torch.utils.data import DataLoader
from useg.wnet.model import WNet
from useg.data import Dataset

import pytorch_lightning as pl


parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation with WNet')
parser.add_argument('--name', metavar='name', default=str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')), type=str,
                    help='Name of model')
                    
parser.add_argument('--squeeze', metavar='K', default=4, type=int, 
                    help='Depth of squeeze layer')
parser.add_argument('--out_Chans', metavar='O', default=3, type=int, 
                    help='Output Channels')
parser.add_argument('--epochs', metavar='e', default=100, type=int, 
                    help='epochs')
parser.add_argument('--input_folder', metavar='f', default=None, type=str, 
                    help='Folder of input images')
parser.add_argument('--output_folder', metavar='of', default=None, type=str, 
                    help='folder of output images')


in_Chans = 1

# Load the arguments
args, unknown = parser.parse_known_args()
wnet = WNet(in_channels=in_Chans, hiddden_dim=8, out_channels=args.squeeze, learning_rate_clust=0.03, learning_rate_recon=0.001, radius=5)

input_files = [
    'C:/Users/federico/Documents/CL/Image15_40.tif', 
    'C:/Users/federico/Documents/CL/Image16_40.tif', 
    'C:/Users/federico/Documents/CL/Image17_40.tif',
    'C:/Users/federico/Documents/CL/Image18_40.tif',
    'C:/Users/federico/Documents/CL/Image19_40.tif',
    'C:/Users/federico/Documents/CL/Image20_40.tif',
    'C:/Users/federico/Documents/CL/Image21_40.tif'
]
dataset = Dataset(input_files)

# Train 1 image set batch size=1 and set shuffle to False
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=args.epochs, log_every_n_steps=1
)

trainer.fit(wnet, dataloader)