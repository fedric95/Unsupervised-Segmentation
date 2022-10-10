from torch.utils.data import DataLoader
from useg.wnet.model import WNet
from useg.wnet.data import Dataset

import pytorch_lightning as pl
import torch
from useg.utils import isimage, getdevice
import os.path

n_clusters = 100
epochs = 100
use_gpu = 1
in_channels = 1
learning_rate_clust = 0.03
learning_rate_recon = 0.001
radius = 5

def transform(out):
    out['image'] = out['image']/255.0
    return(out)

device = getdevice(use_gpu)

# Load the arguments
wnet = WNet(in_channels=in_channels, hiddden_dim=8, intermediate_channels=n_clusters, learning_rate_clust=learning_rate_clust, learning_rate_recon=learning_rate_recon, radius=radius)

input_dir = r'C:/Users/federico/Documents/CL/'
input_files = os.listdir(input_dir)
input_files = [os.path.join(input_dir,file) for file in input_files if isimage(file)]
dataset = Dataset(input_files, transform=transform)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

trainer = pl.Trainer(
    accelerator=device,
    devices=1,
    max_epochs=epochs, log_every_n_steps=1
)

trainer.fit(wnet, dataloader)