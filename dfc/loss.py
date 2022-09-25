import torch.nn as nn
import torch

class CustomLoss(nn.Module):
    
    def __init__(
        self, 
        stepsize_sim,
        stepsize_con,
        stepsize_scr = None,
        scribble = False
    ):
        super(CustomLoss, self).__init__()
        self.stepsize_sim = stepsize_sim
        self.stepsize_con = stepsize_con
        self.scribble = scribble

        # continuity loss definition
        self.loss_hpy = nn.L1Loss(size_average = True)
        self.loss_hpz = nn.L1Loss(size_average = True)

        # similarity loss definition
        self.loss_fn = nn.CrossEntropyLoss()

        # scribble loss definition
        if(self.scribble):
            self.loss_fn_scr = nn.CrossEntropyLoss()
            self.stepsize_scr = stepsize_scr
 
    def forward(
        self,
        HPy,
        HPz,
        output, 
        target,
        inds_sim = None,
        inds_scr = None,
        target_scr = None
    ):
        HPy_target = torch.zeros_like(HPy, requires_grad=False)
        HPz_target = torch.zeros_like(HPz, requires_grad=False)

        if self.scribble:
            l = self.stepsize_sim * self.loss_fn(output[ inds_sim ], target[ inds_sim ])
            l = l + self.stepsize_scr * self.loss_fn_scr(output[ inds_scr ], target_scr[ inds_scr ])
        else:
            l = self.stepsize_sim * self.loss_fn(output, target)
        
        l = l + self.stepsize_con * (self.loss_hpy(HPy,HPy_target) + self.loss_hpz(HPz,HPz_target))

        return(l)