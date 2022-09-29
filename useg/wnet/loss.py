import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_weights(input, batch_size, img_size=(64, 64), ox=4, radius=5 ,oi=10):
    channels = 1
    image = torch.mean(input, dim=1, keepdim=True)
    h, w = img_size
    p = radius

    image = F.pad(input=image, pad=(p, p, p, p), mode='constant', value=0)
    # Use this to generate random values for the padding.
    # randomized_inputs = (0 - 255) * torch.rand(image.shape).cuda() + 255
    # mask = image.eq(0)
    # image = image + (mask *randomized_inputs)

    kh, kw = radius*2 + 1, radius*2 + 1
    dh, dw = 1, 1

    patches = image.unfold(2, kh, dh).unfold(3, kw, dw)

    patches = patches.contiguous().view(batch_size, channels, -1, kh, kw)
    patches = patches.permute(0, 2, 1, 3, 4)
    patches = patches.view(-1, channels, kh, kw)

    center_values = patches[:, :, radius, radius]
    center_values = center_values[:, :, None, None]
    center_values = center_values.expand(-1, -1, kh, kw)

    k_row = (torch.arange(1, kh + 1) - torch.arange(1, kh + 1)[radius]).expand(kh, kw)

    if torch.cuda.is_available():
        k_row = k_row.cuda()

    distance_weights = (k_row ** 2 + k_row.T**2)

    mask = distance_weights.le(radius)
    distance_weights = torch.exp(torch.div(-1*(distance_weights), ox**2))
    distance_weights = torch.mul(mask, distance_weights)


    patches = torch.exp(torch.div(-1*((patches - center_values)**2), oi**2))
    return torch.mul(patches, distance_weights)



def soft_n_cut_loss_single_k(weights, enc, batch_size, img_size, radius=5):
    channels = 1
    h, w = img_size
    p = radius

    kh, kw = radius*2 + 1, radius*2 + 1
    dh, dw = 1, 1
    encoding = F.pad(input=enc, pad=(p, p, p, p), mode='constant', value=0)

    seg = encoding.unfold(2, kh, dh).unfold(3, kw, dw)
    seg = seg.contiguous().view(batch_size, channels, -1, kh, kw)
    seg = seg.permute(0, 2, 1, 3, 4)
    seg = seg.view(-1, channels, kh, kw)

    nom = weights * seg

    nominator = torch.sum(enc * torch.sum(nom, dim=(1,2,3)).reshape(batch_size, h, w), dim=(1,2,3))
    denominator = torch.sum(enc * torch.sum(weights, dim=(1,2,3)).reshape(batch_size, h, w), dim=(1,2,3))

    return torch.div(nominator, denominator)



class SoftCutLoss(nn.Module):
    
    def __init__(self, radius):
        super(SoftCutLoss, self).__init__()
        self.radius = radius

    def forward(self, batch, encoded):
        batch_size = batch.shape[0]
        img_size = [batch.shape[-2], batch.shape[-1]]
        
        loss = []
        k = encoded.shape[1]
        weights = calculate_weights(batch, batch_size, img_size, radius = self.radius)
        for i in range(0, k):
            loss.append(soft_n_cut_loss_single_k(weights, encoded[:, (i,), :, :], batch_size, img_size, radius = self.radius))
        da = torch.stack(loss)

        return(torch.mean(k - torch.sum(da, dim=0)))

        


        