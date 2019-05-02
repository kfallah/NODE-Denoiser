import torch
import torch.nn as nn
import torch.nn.init as init

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

class DnCNN(nn.Module):
    
    def __init__(self, channels=1, num_of_layers=17, kernel_size=3, padding=1, features=64):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        initialize_weights(self)
        
    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, features, droprate=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(features)
        self.nonlin = nn.ReLU(inplace=True)
        self.droprate = droprate
        if self.droprate is not None:
            self.drop = nn.Dropout2d(p=self.droprate)
        initialize_weights(self)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm(out)
        out = self.nonlin(out)
        out = self.conv2(out)
        if self.droprate is not None:
            out = self.drop(out)

        res_out = out + residual

        return res_out                    
                
class ResDnCNN(nn.Module):
    
    def __init__(self, channels=1, num_of_layers=17, kernel_size=3, padding=1, features=64):
        super(ResDnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(BasicBlock(features, droprate=None))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        initialize_weights(self)
        
    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out