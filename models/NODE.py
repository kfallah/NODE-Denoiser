import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint

class NODEDenoiser(nn.Module):
    def __init__(self, channels=1, augmented_channels=5, num_of_layers=17, features=64):
        super(NODEDenoiser, self).__init__()
        self.augmented_channels = augmented_channels
        
        up_conv = nn.Conv2d(channels+self.augmented_channels, features, kernel_size=3, padding=1, bias=False)
        relu = nn.ReLU(inplace=True)
        ode = ODEBlock(ODEDenoiseFunc(features), num_of_layers)
        down_conv = nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False)
        
        self._layers = nn.Sequential(up_conv, relu, ode, down_conv)
        
    def forward(self, x):
        aug_size = list(x.shape)
        aug_size[1] = self.augmented_channels
        aug_channels = torch.zeros(*aug_size, dtype=x.dtype, layout=x.layout, device=x.device)
        x_aug = torch.cat([x, aug_channels], 1)
        return self._layers(x_aug)

class ODEBlock(nn.Module):

    def __init__(self, odefunc, num_of_layers):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.linspace(0, 1, num_of_layers-2).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time)
        return out[-1]

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
        
class ODEDenoiseFunc(nn.Module):

    def __init__(self, features=64):
        super(ODEDenoiseFunc, self).__init__()
        self.conv1 = ConcatConv2d(dim_in=features, dim_out=features, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(features)
        self.nonlin = nn.ReLU(inplace=True)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.conv1(t, x)
        out = self.norm(out)
        out = self.nonlin(out)
        return out
