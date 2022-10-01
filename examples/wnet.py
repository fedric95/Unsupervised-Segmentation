from torch.utils.data import DataLoader
from useg.wnet.model import WNet
from useg.wnet.data import Dataset

import pytorch_lightning as pl
import torch


n_clusters = 100
epochs = 100
use_gpu = 1
in_channels = 1
learning_rate_clust = 0.03
learning_rate_recon = 0.001
radius = 5
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



# Load the arguments
wnet = WNet(in_channels=in_channels, hiddden_dim=8, intermediate_channels=n_clusters, learning_rate_clust=learning_rate_clust, learning_rate_recon=learning_rate_recon, radius=radius)

input_files = [
    'C:/Users/federico/Documents/CL/Image15_40.tif', 
    'C:/Users/federico/Documents/CL/Image16_40.tif', 
    'C:/Users/federico/Documents/CL/Image17_40.tif',
    'C:/Users/federico/Documents/CL/Image18_40.tif',
    'C:/Users/federico/Documents/CL/Image19_40.tif',
    'C:/Users/federico/Documents/CL/Image20_40.tif',
    'C:/Users/federico/Documents/CL/Image21_40.tif'
]
dataset = Dataset(input_files, transform=transform)

# Train 1 image set batch size=1 and set shuffle to False
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


trainer = pl.Trainer(
    accelerator=device,
    devices=1,
    max_epochs=epochs, log_every_n_steps=1
)

trainer.fit(wnet, dataloader)