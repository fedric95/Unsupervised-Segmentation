import torch.cuda
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from neptune.new.types import File

def isimage(path):
    return(path.endswith('.png') or path.endswith('.jpg') or path.endswith('tif'))

def getdevice(use_gpu):
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
    return(device)

def logimage(logger, name, image, global_step):
    if(isinstance(logger, NeptuneLogger)):
        logger.experiment[name].log(File.as_image(image))
    elif(isinstance(logger, TensorBoardLogger)):
        logger.experiment.add_image(name, image, global_step = global_step, dataformats='HWC')