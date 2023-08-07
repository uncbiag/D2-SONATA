
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from Learning.Modules.utils import *

'''Ref: https://github.com/milesial/Pytorch-UNet'''



###################################################################
###################################################################

        
class UNet2D_32(nn.Module):
    def __init__(self, args, in_channels=2, out_channels=1, device = 'cpu', bilinear=True):
        super(UNet2D_32, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 32) # data_dim = 32
        self.down1 = Down(32, 64) # data_dim / 2 = 16
        self.down2 = Down(64, 128) # data_dim / 4 = 8
        self.down3 = Down(128, 256) # data_dim / 8 = 4
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor) # data_dim / 16 = 2
        self.up1 = Up(512, 256 // factor, bilinear) # data_dim / 8 = 4
        self.up2 = Up(256, 128 // factor, bilinear) # data_dim / 4 = 8
        self.up3 = Up(128, 64 // factor, bilinear) # data_dim / 2 = 16
        self.up4 = Up(64, 32, bilinear) # data_dim = 32
        self.outc = OutConv(32, out_channels) # data_dim = 32
        
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            uniform_init(m)
        '''for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)''' 

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) 
        logits = self.outc(x)
        return [logits]


'''class UNet2D_64(nn.Module):
    def __init__(self, args, in_channels=2, out_channels=1, device = 'cpu', bilinear=True):
        super(UNet2D_64, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64) # data_dim = 64
        self.down1 = Down(64, 128)  # data_dim / 2 = 32
        self.down2 = Down(128, 256) # data_dim / 4 = 16
        self.down3 = Down(256, 512) # data_dim / 8 = 8
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor) # data_dim / 16 = 4
        self.up1 = Up(1024, 512 // factor, bilinear) # data_dim / 8 = 8
        self.up2 = Up(512, 256 // factor, bilinear) # data_dim / 4 = 16
        self.up3 = Up(256, 128 // factor, bilinear) # data_dim / 2 = 32
        self.up4 = Up(128, 64, bilinear) # data_dim = 64
        self.outc = OutConv(64, out_channels) # data_dim = 64
        #self.actv = nn.Sigmoid()
        
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            uniform_init(m)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return [logits]
'''

###################################################################
###################################################################


class UNet2D_64(nn.Module):
    def __init__(self, args, in_channels=2, out_channels=1, device = 'cpu', bilinear=True):
        super(UNet2D_64, self).__init__()
        self.in_channels = in_channels 
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64) # data_dim = 64
        self.down1 = Down(64, 128)  # data_dim / 2 = 32
        self.down2 = Down(128, 256) # data_dim / 4 = 16
        self.down3 = Down(256, 512) # data_dim / 8 = 8
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor) # data_dim / 16 = 4
        self.up1 = Up(1024, 512 // factor, bilinear) # data_dim / 8 = 8
        self.up2 = Up(512, 256 // factor, bilinear) # data_dim / 4 = 16
        self.up3 = Up(256, 128 // factor, bilinear) # data_dim / 2 = 32
        self.up4 = Up(128, 64, bilinear) # data_dim = 64
        self.outc = OutConv(64, out_channels) # data_dim = 64
        self.actv = nn.Sigmoid()
        
        self.weight_init()        
        print('=========== Segmentation Net ===========') 

    def weight_init(self):
        #for m in self._modules:
        #    kaiming_init(m) 
        #    zero_init(m)
        for m in self._modules:
            uniform_init(m)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) 
        x = self.actv(self.outc(x))
        return x


class UNet2D_64_VM_SDE(nn.Module):
    def __init__(self, args, in_channels=2, device = 'cpu', bilinear=True):
        super(UNet2D_64_VM_SDE, self).__init__()
        self.in_channels = in_channels 
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64) # data_dim = 64
        self.down1 = Down(64, 128)  # data_dim / 2 = 32
        self.down2 = Down(128, 256) # data_dim / 4 = 16
        self.down3 = Down(256, 512) # data_dim / 8 = 8
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor) # data_dim / 16 = 4

        self.up1_vm = Up(1024, 512 // factor, bilinear) # data_dim / 8 = 8
        self.up2_vm = Up(512, 256 // factor, bilinear) # data_dim / 4 = 16
        self.up3_vm = Up(256, 128 // factor, bilinear) # data_dim / 2 = 32
        self.up4_vm = Up(128, 64, bilinear) # data_dim = 64
        self.outc_vm = OutConv(64, 1) # data_dim = 64
        self.actv_vm = nn.Sigmoid()

        self.up1_s = Up(1024, 512 // factor, bilinear) # data_dim / 8 = 8
        self.up2_s = Up(512, 256 // factor, bilinear) # data_dim / 4 = 16
        self.up3_s = Up(256, 128 // factor, bilinear) # data_dim / 2 = 32
        self.up4_s = Up(128, 64, bilinear) # data_dim = 64
        self.outc_s = OutConv(64, 1) # data_dim = 64
        self.actv_s = nn.Sigmoid()
        
        self.weight_init()        
        print('=========== Segmentation Net ===========') 

    def weight_init(self):
        #for m in self._modules:
        #    kaiming_init(m) 
        #    zero_init(m)
        for m in self._modules:
            uniform_init(m)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_vm = self.up1_vm(x5, x4)
        x_vm = self.up2_vm(x_vm, x3)
        x_vm = self.up3_vm(x_vm, x2)
        x_vm = self.up4_vm(x_vm, x1) 
        x_vm = self.actv_vm(self.outc_vm(x_vm))

        x_s = self.up1_s(x5, x4)
        x_s = self.up2_s(x_s, x3)
        x_s = self.up3_s(x_s, x2)
        x_s = self.up4_s(x_s, x1) 
        x_s = self.actv_s(self.outc_s(x_s))

        return x_vm, x_s



###################################################################
###################################################################


class UNet2D_64_SplitDecoder(nn.Module):
    def __init__(self, args, in_channels=2, out_channels=1, trilinear = False, SepActivate = None, stochastic = False):
        super(UNet2D_64_SplitDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear
        self.SepActivate = SepActivate
        self.stochastic = stochastic 
        self.stochastic_separate_net = args.stochastic_separate_net
        self.predict_value_mask = args.predict_value_mask 
        self.separate_DV_value_mask = args.separate_DV_value_mask
        self.value_mask_separate_net = args.value_mask_separate_net 
        self.is_vm_sde_net = args.vm_sde_net
        self.predict_deviation = args.predict_deviation and len(SepActivate) > 2 
 
        factor = 2 if self.trilinear else 1
        if SepActivate is None:
            raise ValueError('SepActivate channels must be indicated.')

        print('=========== Register UNet2D_64: Uniform_init (Joint Encoder, Splited Decoders) ===========')
                     
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1_v = Up(1024, 512 // factor, self.trilinear)
        self.up2_v = Up(512, 256 // factor, self.trilinear)
        self.up3_v = Up(256, 128 // factor, self.trilinear)
        self.up4_v = Up(128, 64, self.trilinear)
        self.outcV = OutConv(64, self.SepActivate[0])

        self.up1_d = Up(1024, 512 // factor, self.trilinear)
        self.up2_d = Up(512, 256 // factor, self.trilinear)
        self.up3_d = Up(256, 128 // factor, self.trilinear)
        self.up4_d = Up(128, 64, self.trilinear)
        self.outcD = OutConv(64, self.SepActivate[1])

        if self.predict_deviation:
            print('===========                         Predict Deviations                         ===========')
            self.up1_dv = Up(1024, 512 // factor, self.trilinear)
            self.up2_dv = Up(512, 256 // factor, self.trilinear)
            self.up3_dv = Up(256, 128 // factor, self.trilinear)
            self.up4_dv = Up(128, 64, self.trilinear)
            self.outcdV = OutConv(64, self.SepActivate[0])

            self.up1_dd = Up(1024, 512 // factor, self.trilinear)
            self.up2_dd = Up(512, 256 // factor, self.trilinear)
            self.up3_dd = Up(256, 128 // factor, self.trilinear)
            self.up4_dd = Up(128, 64, self.trilinear)
            self.outcdD = OutConv(64, self.SepActivate[2]) 

        if self.is_vm_sde_net: # NOTE: archived, not work well
            assert args.predict_value_mask and args.stochastic
            self.vm_sde_net = UNet2D_64_VM_SDE(args, in_channels)
        else:
            if self.predict_value_mask:
                if not self.value_mask_separate_net:
                    print('===========                          (Value Mask Net)                          ===========')
                    self.up1_vm = Up(1024, 512 // factor, self.trilinear)
                    self.up2_vm = Up(512, 256 // factor, self.trilinear)
                    self.up3_vm = Up(256, 128 // factor, self.trilinear)
                    self.up4_vm = Up(128, 64, self.trilinear)

                    #########################
                    ## NOTE: To be determined
                    #########################
                    if self.separate_DV_value_mask: 
                        self.outcVM = OutConv(64, 2)
                    else:
                        self.outcVM = OutConv(64, 1)
                    #########################
                    #########################

                    self.actvVM = nn.Sigmoid()  
                else:
                    if self.separate_DV_value_mask:
                        self.value_mask_net = UNet2D_64(args, in_channels, out_channels = 2)
                    else:
                        self.value_mask_net = UNet2D_64(args, in_channels, out_channels = 1)

            if self.stochastic: 
                if not self.stochastic_separate_net:
                    print('===========                     (Stochatic Version (Joint))                    ===========')
                    self.up1_s = Up(1024, 512 // factor, self.trilinear)
                    self.up2_s = Up(512, 256 // factor, self.trilinear)
                    self.up3_s = Up(256, 128 // factor, self.trilinear)
                    self.up4_s = Up(128, 64, self.trilinear)
                    #if self.separate_DV_value_mask:
                    #    self.outcS = OutConv(64, 2)  
                    #else:
                    #    self.outcS = OutConv(64, 1)  
                    self.outcS = OutConv(64, 1) 
                    self.actvS = nn.Sigmoid()
                else:
                    self.stochastic_net = UNet2D_64(args, in_channels, out_channels = 1) 
        
        self.weight_init()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x_v = self.up1_v(x5, x4)
        x_v = self.up2_v(x_v, x3)
        x_v = self.up3_v(x_v, x2)
        x_v = self.up4_v(x_v, x1) 
        
        x_d = self.up1_d(x5, x4)
        x_d = self.up2_d(x_d, x3)
        x_d = self.up3_d(x_d, x2)
        x_d = self.up4_d(x_d, x1) 

        if self.is_vm_sde_net: # NOTE: archived, not work well
            x_vm, x_s = self.vm_sde_net(x)
            out = torch.cat([self.outcV(x_v), self.outcD(x_d)], dim = 1) 
        else:
            if self.stochastic:
                if self.stochastic_separate_net:
                    x_s = self.stochastic_net(x)
                    out = torch.cat([self.outcV(x_v), self.outcD(x_d)], dim = 1)  
                else:
                    x_s = self.up1_s(x5, x4)
                    x_s = self.up2_s(x_s, x3)
                    x_s = self.up3_s(x_s, x2)
                    x_s = self.up4_s(x_s, x1) 
                    x_s = self.actvS(self.outcS(x_s))
                    out = torch.cat([self.outcV(x_v), self.outcD(x_d)], dim = 1)
            else:
                out = torch.cat([self.outcV(x_v), self.outcD(x_d)], dim = 1)
                x_s = None
            
            if self.predict_value_mask:
                if self.value_mask_separate_net: # NOTE: archived, not work well
                    x_vm = self.value_mask_net(x)
                else:
                    x_vm = self.up1_vm(x5, x4)
                    x_vm = self.up2_vm(x_vm, x3)
                    x_vm = self.up3_vm(x_vm, x2)
                    x_vm = self.up4_vm(x_vm, x1)  
                    x_vm = self.actvVM(self.outcVM(x_vm)) # (n_batch, 1, r, c) or (n_batch, 2, r, c)
            else:
                x_vm = None

        if self.predict_deviation:
            x_dv = self.up1_dv(x5, x4)
            x_dv = self.up2_dv(x_dv, x3)
            x_dv = self.up3_dv(x_dv, x2)
            x_dv = self.up4_dv(x_dv, x1) 

            x_dd = self.up1_dd(x5, x4)
            x_dd = self.up2_dd(x_dd, x3)
            x_dd = self.up3_dd(x_dd, x2)
            x_dd = self.up4_dd(x_dd, x1) 

            delta_out = torch.cat([self.outcdV(x_dv), self.outcdD(x_dd)], dim = 1)
        else:
            delta_out = None

        return out, delta_out, x_vm, x_s

    def weight_init(self):
        #for m in self._modules:
        #    kaiming_init(m)
        #for m in self._modules:
        #    zero_init(m)
        for m in self._modules:
            uniform_init(m)
        '''for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)'''


class UNet2D_64_JointDecoder(nn.Module):
    def __init__(self, args, in_channels = 2, out_channels = 1, trilinear = False, SepActivate = None):
        super(UNet2D_64_JointDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear
        self.SepActivate = SepActivate
        self.stochastic = args.stochastic
        self.separate_stochastic = args.separate_stochastic
        self.predict_deviation = args.predict_deviation
        factor = 2 if self.trilinear else 1

        print('===========     Register UNet2D_64: Uniform_init (Joint Encoder & Decoder)     ===========')

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.trilinear)
        self.up2 = Up(512, 256 // factor, self.trilinear)
        self.up3 = Up(256, 128 // factor, self.trilinear)
        self.up4 = Up(128, 64, self.trilinear)

        if self.SepActivate: # Seperate output activatioon layer #
            self.outcV = OutConv(64, self.SepActivate[0])
            self.outcD = OutConv(64, self.SepActivate[1])
        else:
            self.outc = OutConv(64, out_channels)

        if self.predict_deviation:
            print('===========                         Predict Deviations                         ===========')
            self.outcdV = OutConv(64, self.SepActivate[0])
            self.outcdD = OutConv(64, self.SepActivate[2])

        if self.stochastic:
            if not self.separate_stochastic:
                print('===========                     (Stochatic Version (Joint))                    ===========')
                self.outcS = OutConv(64, 1)
            elif self.SepActivate:
                print('===========                     (Stochatic Version (Split))                    ===========')
                self.outcS_V = OutConv(64, 1)
                self.outcS_D = OutConv(64, 1)
            else:
                raise ValueError('Does not support separate stochastic option when not SepActivate')
        
        self.weight_init()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) 

        if self.SepActivate:
            out = torch.cat([self.outcV(x), self.outcD(x)], dim = 1)
        else:
            out = self.outc(x)

        if self.stochastic:
            if not self.separate_stochastic:
                out = torch.cat([out, self.outcS(x)], dim = 1)
            else:
                out = torch.cat([out, self.outcS_V(x), self.outcS_D(x)], dim = 1)

        if self.predict_deviation:
            delta_out = torch.cat([self.outcdV(x), self.outcdD(x)], dim = 1)
        else:
            delta_out = None

        return out, delta_out

    def weight_init(self):
        #for m in self._modules:
        #    kaiming_init(m)
        #for m in self._modules:
        #    zero_init(m)
        for m in self._modules:
            uniform_init(m)
        '''for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)'''
                

###########################################################
########################## Utils ##########################
###########################################################


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)