#from __future__ import print_function
import argparse
import torch
import torch.nn.init
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import Net
from loss import CustomLoss
from data import Dataset


parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False, 
                    help='use scribbles')
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
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float, 
                    help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float, 
                    help='step size for scribble loss')
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


scribble_file = None
if args.scribble:
    scribble_file = args.input.replace('.'+args.input.split('.')[-1],'_scribble.png')

input_files = []

scribble_files = None

dataset = Dataset(input_files, scribble_files)

in_channels = 1
custom_loss = CustomLoss(stepsize_sim = args.stepsize_sim, stepsize_con = args.stepsize_con, stepsize_scr = args.stepsize_scr, scribble=args.scribble)
model = Net( 
    in_channels = in_channels, 
    out_channels = args.nChannel, 
    n_layers = args.nConv, 
    loss=custom_loss,
    learning_rate=args.lr
)

trainer = pl.Trainer(
    accelerator=device,
    devices=1,
    max_epochs=args.maxIter, log_every_n_steps=1
)
trainer.fit(model, DataLoader(dataset, batch_size = None))