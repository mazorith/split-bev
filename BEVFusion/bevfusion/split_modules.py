import copy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mmdet3d.registry import MODELS

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN1

@MODELS.register_module()
class SplitLidarEncoder(nn.Module):
    def __init__(self, final_feature_size: int, target_input_size: int, out_channels: int) -> None:
        super().__init__()
        self.feature_size = final_feature_size
        self.input_size = target_input_size
        self.out_channels = out_channels
        
        #currently expecting lidar lists to be around ~287496 is points so
        #I will hardcode for that a the moment 

        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 1, 1, stride = 1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
        )

        self.linear1 = nn.Sequential(
            nn.Linear(10000, 1000, bias=False),
            nn.LeakyReLU(True),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(28000, 2025, bias=False),
            nn.LeakyReLU(True),
        )

        self.train_recon = True

        if self.train_recon:
            self.conv2 = nn.Sequential(
                nn.Conv2d(1, 5, 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm2d(5),
                nn.ReLU(True),
            )

            self.trans_conv1 = nn.Sequential(
                nn.ConvTranspose2d(5, 25, 2, stride = 2, padding=0, bias=False),
                nn.BatchNorm2d(25),
                nn.ReLU(True),
            )

            self.conv3 = nn.Sequential(
                nn.Conv2d(25, 75, 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm2d(75),
                nn.ReLU(True),
            )

            self.trans_conv2 = nn.Sequential(
                nn.ConvTranspose2d(75, 175, 2, stride = 2, padding=0, bias=False),
                nn.BatchNorm2d(175),
                nn.ReLU(True),
            )

            self.conv4 = nn.Sequential(
                nn.Conv2d(175, 256, 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )

            self.conv5 = nn.Sequential(
                nn.Conv2d(256, 256, 3, stride = 1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )

            

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.conv1(inputs)
        inputs = inputs.reshape(inputs.shape[0],-1)

        interm = None
        token = None
        for i in range(0, 280000, 10000):
            if interm == None:
                interm = self.linear1(inputs[:,i:i+10000])
            else:
                if token == None:
                    token = self.linear1(inputs[:,i:i+10000]) + (interm*0.02)
                else:
                    token = self.linear1(inputs[:,i:i+10000]) + (token*0.02)
                interm = torch.cat((interm, token), dim=1)
            
        inputs = self.linear2(interm)
        
        inputs = inputs.reshape(inputs.shape[0],1,45,45)
        if self.train_recon:
            inputs = self.conv2(inputs)
            inputs = self.trans_conv1(inputs)
            inputs = self.conv3(inputs)
            inputs = self.trans_conv2(inputs)
            inputs = self.conv4(inputs)
            inputs = self.conv5(inputs)

        return inputs
    
@MODELS.register_module()
class SplitImageEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        #currently expecting lidar lists to be around ~287496 is points so
        #I will hardcode for that a the moment 

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 100, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(100),
            nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(100, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.conv3(inputs)
        # inputs = inputs.view(B, int(BN / B), C, H, W)
        return inputs
    

@MODELS.register_module()
class FuserNeck(nn.Module):
    #mid_channels should end up being a list
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        super().__init__()

        self.fuse_layer = nn.Sequential(
            nn.Conv2d(336, 200, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(200),
            nn.ReLU(True),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv2d(200, 100, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(100),
            nn.ReLU(True),
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(100, 50, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(50),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(inputs, dim=1)

        x = self.fuse_layer(x)
        x = self.down_conv1(x)
        x = self.down_conv2(x)

        return x
    

@MODELS.register_module()
class FuserNeck_entro(nn.Module):
    #mid_channels should end up being a list
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        super().__init__()

        self.fuse_layer = nn.Sequential(
            nn.Conv2d(336, 200, 3, stride=2, padding=1, bias=False),
            GDN1(200)
            # nn.BatchNorm2d(200),
            # nn.ReLU(True),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv2d(200, 100, 3, stride=2, padding=1, bias=False),
            GDN1(100)
            # nn.BatchNorm2d(100),
            # nn.ReLU(True),
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(100, 50, 3, stride=1, padding=1, bias=False),
            GDN1(50)
            # nn.BatchNorm2d(50),
            # nn.ReLU(True),
        )

        self.entropy_bb = EntropyBottleneck(50)

        self.stage = 0

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(inputs, dim=1)

        x = self.fuse_layer(x)
        x = self.down_conv1(x)
        x = self.down_conv2(x)

        if self.stage != 0:
            size = [x.shape[2], x.shape[3]]
            y_hat = self.entropy_bb.compress(x)
            y_likelihoods = size

            #currently not physically spliting the model so this is ok for now
            y_hat = self.entropy_bb.decompress(y_hat, size)
        else:   
            y_hat, y_likelihoods = self.entropy_bb(x, (self.stage==0))

        return (y_hat, y_likelihoods) 
    

@MODELS.register_module()
class FuserDecoder(nn.Module):
    #mid_channels should end up being a list
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        super().__init__()

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(50, 90, 2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(90),
            nn.ReLU(True),
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(90, 180, 2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(180),
            nn.ReLU(True),
        )

        self.set_conv1 = nn.Sequential(
            nn.Conv2d(180, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.set_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        inputs = self.up_conv1(inputs)
        inputs = self.up_conv2(inputs)
        inputs = self.set_conv1(inputs)
        inputs = self.set_conv2(inputs)
    
        return inputs
    

@MODELS.register_module()
class FuserDecoder_entro(nn.Module):
    #mid_channels should end up being a list
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        super().__init__()

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(50, 90, 2, stride=2, padding=0, bias=False),
            GDN1(90, inverse=True)
            # nn.BatchNorm2d(90),
            # nn.ReLU(True),
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(90, 180, 2, stride=2, padding=0, bias=False),
            GDN1(180, inverse=True)
            # nn.BatchNorm2d(180),
            # nn.ReLU(True),
        )

        self.set_conv1 = nn.Sequential(
            nn.Conv2d(180, 256, 3, stride=1, padding=1, bias=False),
            GDN1(256, inverse=True)
            # nn.BatchNorm2d(256),
            # nn.ReLU(True),
        )

        self.set_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        inputs = self.up_conv1(inputs)
        inputs = self.up_conv2(inputs)
        inputs = self.set_conv1(inputs)
        inputs = self.set_conv2(inputs)
    
        return inputs