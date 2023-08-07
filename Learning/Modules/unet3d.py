
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from Learning.Modules.utils import *
from Learning.Modules.AdvDiffPDE import *
#from Learning.Modules.unet3d_old import UNet3D

'''https://github.com/milesial/Pytorch-UNet'''




###################################################################
###################################################################


class UNet3D_32_Shallow(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, trilinear = False, SepActivate = None):
        super(UNet3D_32_Shallow, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear
        self.SepActivate = SepActivate
        factor = 2 if self.trilinear else 1

        print('=========== Register UNet3D_32_Shallow ===========')

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, self.trilinear)
        self.up2 = Up(128, 64 // factor, self.trilinear)
        self.up3 = Up(64, 32, self.trilinear)
        if self.SepActivate:
            self.outcV = OutConv(32, self.SepActivate[0])
            self.outcD = OutConv(32, self.SepActivate[1])
        else:
            self.outc = OutConv(32, out_channels)
        
        self.weight_init()

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

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        if self.SepActivate:
            x = torch.cat([self.outcV(x), self.outcD(x)], dim = 1)
        else:
            x = self.outc(x)
        return x


###################################################################
###################################################################


class UNet3D_32(nn.Module):
    def __init__(self, args, in_channels=2, out_channels=1, trilinear = False):
        super(UNet3D_32, self).__init__()
        self.in_channels = in_channels 
        self.trilinear = trilinear 
        factor = 2 if self.trilinear else 1

        print('=========== Segmentation Net ===========') 

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512 // factor)

        self.up1 = Up(512, 256 // factor, self.trilinear)
        self.up2 = Up(256, 128 // factor, self.trilinear)
        self.up3 = Up(128, 64 // factor, self.trilinear)
        self.up4 = Up(64, 32, self.trilinear)
        self.outc = OutConv(32, out_channels) # data_dim = 64
        self.actv = nn.Sigmoid() 
        
        self.weight_init()

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



class UNet3D_32_VM_SDE(nn.Module):
    def __init__(self, args, in_channels=2, trilinear = False):
        super(UNet3D_32_VM_SDE, self).__init__()
        self.in_channels = in_channels 
        self.trilinear = trilinear 
        factor = 2 if self.trilinear else 1

        print('=========== ValueMask & Stochastic Net ===========') 

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512 // factor)
    
        self.up1_vm = Up(512, 256 // factor, self.trilinear)
        self.up2_vm = Up(256, 128 // factor, self.trilinear)
        self.up3_vm = Up(128, 64 // factor, self.trilinear)
        self.up4_vm = Up(64, 32, self.trilinear)
        self.outc_vm = OutConv(32, 1) # data_dim = 64
        self.actv_vm = nn.Sigmoid() 

        self.up1_s = Up(512, 256 // factor, self.trilinear)
        self.up2_s = Up(256, 128 // factor, self.trilinear)
        self.up3_s = Up(128, 64 // factor, self.trilinear)
        self.up4_s = Up(64, 32, self.trilinear)
        self.outc_s = OutConv(32, 1) # data_dim = 64
        self.actv_s = nn.Sigmoid() 
            
        self.weight_init()

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

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        vm = self.up1_vm(x5, x4)
        vm = self.up2_vm(vm, x3)
        vm = self.up3_vm(vm, x2)
        vm = self.up4_vm(vm, x1)
        vm = self.actv_vm(self.outc_vm(vm))

        s = self.up1_s(x5, x4)
        s = self.up2_s(s, x3)
        s = self.up3_s(s, x2)
        s = self.up4_s(s, x1)
        s = self.actv_s(self.outc_s(s))
        return vm, s


###################################################################
###################################################################


class UNet3D_32_SplitDecoder(nn.Module):
    def __init__(self, args, in_channels=2, out_channels=1, trilinear = False, SepActivate = None, stochastic = False):
        super(UNet3D_32_SplitDecoder, self).__init__()
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

        print('=========== Register UNet3D_32: Uniform_init (Splited Encoder, Splited Decoders) ===========')

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512 // factor)

        self.up1_v = Up(512, 256 // factor, self.trilinear)
        self.up2_v = Up(256, 128 // factor, self.trilinear)
        self.up3_v = Up(128, 64 // factor, self.trilinear)
        self.up4_v = Up(64, 32, self.trilinear)
        self.outcV = OutConv(32, self.SepActivate[0])

        self.up1_d = Up(512, 256 // factor, self.trilinear)
        self.up2_d = Up(256, 128 // factor, self.trilinear)
        self.up3_d = Up(128, 64 // factor, self.trilinear)
        self.up4_d = Up(64, 32, self.trilinear)
        self.outcD = OutConv(32, self.SepActivate[1])

        if self.predict_deviation: # NOTE: archived -- not work well
            print('===========                         Predict Deviations                         ===========')
            self.up1_dv = Up(512, 256 // factor, self.trilinear)
            self.up2_dv = Up(256, 128 // factor, self.trilinear)
            self.up3_dv = Up(128, 64 // factor, self.trilinear)
            self.up4_dv = Up(64, 32, self.trilinear)
            self.outcdV = OutConv(32, self.SepActivate[0])

            self.up1_dd = Up(512, 256 // factor, self.trilinear)
            self.up2_dd = Up(256, 128 // factor, self.trilinear)
            self.up3_dd = Up(128, 64 // factor, self.trilinear)
            self.up4_dd = Up(64, 32, self.trilinear)
            self.outcdD = OutConv(32, self.SepActivate[2]) 

        if self.is_vm_sde_net:
            assert args.predict_value_mask and args.stochastic
            self.vm_sde_net = UNet3D_32_VM_SDE(args, in_channels)
        else:
            if self.predict_value_mask: 
                if not self.value_mask_separate_net:
                    print('===========                          (Value Mask Net)                          ===========')
                    self.up1_vm = Up(512, 256 // factor, self.trilinear)
                    self.up2_vm = Up(256, 128 // factor, self.trilinear)
                    self.up3_vm = Up(128, 64 // factor, self.trilinear)
                    self.up4_vm = Up(64, 32, self.trilinear)

                    ##########################
                    # NOTE: to be determined #
                    ##########################
                    if self.separate_DV_value_mask:
                        self.outcVM = OutConv(32, 2)
                    else:
                        self.outcVM = OutConv(32, 1)
                    ##########################
                    
                    self.actvVM = nn.Sigmoid() 
                else:
                    self.value_mask_net = UNet3D_32(args, in_channels, out_channels = 2) if self.separate_DV_value_mask else UNet2D_32(args, in_channels, out_channels = 1)

            if self.stochastic: 
                if not self.stochastic_separate_net:
                    print('===========                     (Stochatic Version (Joint))                    ===========')
                    self.up1_s = Up(512, 256 // factor, self.trilinear)
                    self.up2_s = Up(256, 128 // factor, self.trilinear)
                    self.up3_s = Up(128, 64 // factor, self.trilinear)
                    self.up4_s = Up(64, 32, self.trilinear)
                    self.outcS = OutConv(32, 1)   
                    #if self.separate_DV_value_mask:
                    #    self.outcS = OutConv(32, 2)  
                    #else:
                    #    self.outcS = OutConv(32, 1)   
                    self.actvS = nn.Sigmoid() 
                else:
                    self.stochastic_net = UNet3D_32(args, in_channels)
        
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

        if self.is_vm_sde_net:
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
                if self.value_mask_separate_net:
                    x_vm = self.value_mask_net(x)
                else:
                    x_vm = self.up1_vm(x5, x4)
                    x_vm = self.up2_vm(x_vm, x3)
                    x_vm = self.up3_vm(x_vm, x2)
                    x_vm = self.up4_vm(x_vm, x1)   
                    x_vm = self.actvVM(self.outcVM(x_vm)) # (n_batch, 1, s, r, c) or (n_batch, 2, s, r, c) -- depends on whether predicting separate D&V separate anomaly masks #
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


###############################################


class UNet3D_32_JointDecoder(nn.Module):
    def __init__(self, args, in_channels=2, out_channels=1, trilinear = False, SepActivate = None):
        super(UNet3D_32_JointDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear
        self.SepActivate = SepActivate
        self.stochastic = args.stochastic 
        self.predict_deviation = args.predict_deviation

        factor = 2 if self.trilinear else 1
        print('=========== Register UNet3D_32: Uniform_init (Joint Encoder & Decoder) ===========')

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, self.trilinear)
        self.up2 = Up(256, 128 // factor, self.trilinear)
        self.up3 = Up(128, 64 // factor, self.trilinear)
        self.up4 = Up(64, 32, self.trilinear)

        if self.SepActivate: # Seperate output activatioon layer #
            self.outcV = OutConv(32, self.SepActivate[0])
            self.outcD = OutConv(32, self.SepActivate[1])
        else:
            self.outc = OutConv(32, out_channels)

        if self.predict_deviation:
            print('===========                         Predict Deviations                         ===========')
            self.outcdV = OutConv(32, self.SepActivate[0])
            self.outcdD = OutConv(32, self.SepActivate[2])

        if self.stochastic:  
            print('===========                     (Stochatic Version (Joint))                    ===========')
            self.outcS = OutConv(32, 1)  
        
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
            out = torch.cat([out, self.outcS(x)], dim = 1) 

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


###################################################################
###################################################################


class UNet3D_VesselAttent_32(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, trilinear = False, SepActivate = None):
        super(UNet3D_VesselAttent_32, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear
        self.SepActivate = SepActivate
        factor = 2 if self.trilinear else 1 

        print('=========== Register UNet3D_VesselAttent_32 ===========')

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512 // factor)

        self.up1 = Up(512, 256 // factor, self.trilinear)
        self.up2 = Up(256, 128 // factor, self.trilinear)
        self.up3 = Up(128, 64 // factor, self.trilinear)
        self.up4 = Up(64, 32, self.trilinear)
        if self.SepActivate:
            self.outcV = OutConv(32, self.SepActivate[0])
            self.outcD = OutConv(32, self.SepActivate[1])
        else:
            self.outc = OutConv(32, out_channels)

        self.up1_attent = Up(512, 256 // factor, self.trilinear)
        self.up2_attent = Up(256, 128 // factor, self.trilinear)
        self.up3_attent = Up(128, 64 // factor, self.trilinear)
        self.up4_attent = Up(64, 32, self.trilinear)
        self.out_attent = nn.Sequential(
            OutConv(32, 1),
            nn.Sigmoid()
        )
        
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

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

        x_attent = self.up1_attent(x5, x4)
        x_attent = self.up2_attent(x_attent, x3)
        x_attent = self.up3_attent(x_attent, x2)
        x_attent = self.up4_attent(x_attent, x1) 
        x_attent = self.out_attent(x_attent)

        if self.SepActivate:
            x = torch.cat([self.outcV(x), self.outcD(x)], dim = 1)
        else:
            x = self.outc(x)

        return [[x], [x_attent]]


class UNet3D_64(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, trilinear = False, SepActivate = None):
        super(UNet3D_64, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear
        self.SepActivate = SepActivate

        print('=========== Register UNet3D_64 ===========')

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if trilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.trilinear)
        self.up2 = Up(512, 256 // factor, self.trilinear)
        self.up3 = Up(256, 128 // factor, self.trilinear)
        self.up4 = Up(128, 64, self.trilinear)
        if self.SepActivate:
            self.outcV = OutConv(64, self.SepActivate[0])
            self.outcD = OutConv(64, self.SepActivate[1])
        else:
            self.outc = OutConv(64, out_channels)
        
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

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
            x = torch.cat([self.outcV(x), self.outcD(x)], dim = 1)
        else:
            x = self.outc(x)

        return x


class Discriminator3D_32(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, trilinear = False):
        super(Discriminator3D_32, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if trilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, trilinear)
        self.up2 = Up(256, 128 // factor, trilinear)
        self.up3 = Up(128, 64 // factor, trilinear)
        self.up4 = Up(64, 32, trilinear) # data_dim = 32
        self.activation = nn.Linear(1048576, 1) # 32 * 32 * 32 * 32
        self.classifier = nn.Sigmoid()
        
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

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
        x = self.activation(x.view(x.size(0), -1))
        p = self.classifier(x)
        return p


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
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=2, stride=2)
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
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
