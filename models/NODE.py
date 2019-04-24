import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint

class ODEBlock(nn.Module):

    def __init__(self, odefunc, rtol, atol):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.rtol = rtol
        self.atol = atol
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.to(x.device).type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol)
        return out[-1]
    
    def set_tolerance(self, rtol, atol):
        self.rtol = rtol
        self.atol = atol

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
        
class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)
    
class ODEfunc(nn.Module):

    def __init__(self, dim=64):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out        
        
class ODEDenoiseFunc(nn.Module):

    def __init__(self, features=64):
        super(ODEDenoiseFunc, self).__init__()
        self.conv1 = ConcatConv2d(dim_in=features, dim_out=features, padding=1, bias=False)
        self.norm = norm(features)
        self.nonlin = nn.ReLU(inplace=True)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.conv1(t, x)
        out = self.norm(out)
        out = self.nonlin(out)
        return out

class NODEDenoiser(nn.Module):
    def __init__(self, channels=1, func=ODEDenoiseFunc, augmented_channels=0, features=64, rtol=1e-6, atol=1e-12):
        super(NODEDenoiser, self).__init__()
        self.augmented_channels = augmented_channels
        
        up_conv = nn.Conv2d(channels+self.augmented_channels, features, kernel_size=3, padding=1, bias=False)
        relu = nn.ReLU(inplace=True)
        self.ode = ODEBlock(func(features), rtol=rtol, atol=atol)
        down_conv = nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False)
        
        self._layers = nn.Sequential(up_conv, relu, self.ode, down_conv)
        
    def forward(self, x):
        y = x
        
        aug_size = list(x.shape)
        aug_size[1] = self.augmented_channels
        aug_channels = torch.zeros(*aug_size, dtype=x.dtype, layout=x.layout, device=x.device)
        x_aug = torch.cat([x, aug_channels], 1)
        out = self._layers(x_aug)
        
        return y-out
    
    def set_tolerance(self, rtol, atol):
        self.ode.set_tolerance(rtol, atol)