# Interaction Perception
import torch
import torch.nn as nn
from model.layers import Attention as Att, GCN
from model.layers import IPM_IAW
from model.layers import PE
from torchvision.transforms import transforms as tf


class IP(nn.Module):
    """
    Interaction Perceptron Module
    """
    def __init__(self, opt, dim_in, mid_feature, num_axis, dropout=0.1):
        super(IP, self).__init__()
        self.mid_feature = mid_feature
        self.opt = opt

    def forward(self, x, ori_input, num_person, bat_ITW, IAW, Viz= False):
        self.num_person = num_person
        out=[]
        ip_each = []
        for i in range(self.num_person):

            xni = x[:, i, :, :]
            inp = ori_input[:, i, :, :]
            LP_score = bat_ITW[:, i].unsqueeze(1).unsqueeze(2).repeat(1, self.opt.in_features, self.mid_feature)
            AP_score, change = IAW(inp)
          

            if Viz:
                if i ==0:
                    newx_VIZ = torch.mul(torch.mul(xni, AP_score), LP_score)
                    out = newx_VIZ.unsqueeze(1)
                else:
                    out = torch.cat([out, newx_VIZ.unsqueeze(1)], 1) 
            else:
                if i == 0:
                    newx = torch.mul(torch.mul(xni, AP_score), LP_score)
                    ip_each = torch.mul(torch.mul(xni, AP_score), LP_score).unsqueeze(1)
                else:
                    newx += torch.mul(torch.mul(xni, AP_score), LP_score)
                    ip_each = torch.cat([ip_each ,torch.mul(torch.mul(xni, AP_score), LP_score).unsqueeze(1)], 1)
        if Viz:
            return out
        else:    
            x = newx.clone()

            return x,ip_each
