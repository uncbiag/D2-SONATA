import torch
import torch.nn as nn
import torch.nn.functional as F

from Learning.Modules import groupnorm


class UNet3D(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        interpolate (bool): if True use F.interpolate for upsampling otherwise
            use ConvTranspose3d
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `DoubleConv` for more info.
        active_unit (str): 'relu', 'elu', 'leaky_relu'
    """

    def __init__(self, in_channels, out_channels, interpolate = True, conv_layer_order = 'crg',
                 init_channel_number = 32, SepActivate = None, active_unit = 'relu'):

        super(UNet3D, self).__init__()

        # Number of groups for the GroupNorm
        num_groups = min(init_channel_number // 2, 32)

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper 
        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, is_max_pool = False, conv_layer_order = conv_layer_order,
                    num_groups = num_groups, active_unit = active_unit),
            Encoder(init_channel_number, 2 * init_channel_number, conv_layer_order = conv_layer_order,
                    num_groups = num_groups, active_unit = active_unit),
            Encoder(2 * init_channel_number, 4 * init_channel_number, conv_layer_order = conv_layer_order,
                    num_groups = num_groups, active_unit = active_unit),
            Encoder(4 * init_channel_number, 8 * init_channel_number, conv_layer_order = conv_layer_order,
                    num_groups = num_groups, active_unit = active_unit),
            Encoder(8 * init_channel_number, 16 * init_channel_number, conv_layer_order = conv_layer_order,
                    num_groups = num_groups, active_unit = active_unit)
        ])

        self.decoders = nn.ModuleList([
            Decoder(8 * init_channel_number + 16 * init_channel_number, 8 * init_channel_number, interpolate,
                    conv_layer_order = conv_layer_order, num_groups = num_groups, active_unit = active_unit),
            Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number, interpolate,
                    conv_layer_order = conv_layer_order, num_groups = num_groups, active_unit = active_unit),
            Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number, interpolate,
                    conv_layer_order = conv_layer_order, num_groups = num_groups, active_unit = active_unit),
            Decoder(init_channel_number + 2 * init_channel_number, init_channel_number, interpolate,
                    conv_layer_order = conv_layer_order, num_groups = num_groups, active_unit = active_unit)
        ])

        # In the last layer a 1×1×1 convolution reduces # of output channels to # of labels
        self.SepActivate = SepActivate
        if self.SepActivate:
            assert out_channels == self.SepActivate[0] + self.SepActivate[1]
            self.final_conv1 = nn.Conv3d(init_channel_number, self.SepActivate[0], 1)
            self.final_conv2 = nn.Conv3d(init_channel_number, self.SepActivate[1], 1)
        else:
            self.final_conv = nn.Conv3d(init_channel_number, out_channels, 1)



    def forward(self, x):

        # Encoder part

        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # Reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
            #print('Encoder')
            #print(x.size())

        # Remove the last encoder's output from the list
        # Remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # Decoder part

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            #print('Decoder')
            #print(encoder_features.size())
            x = decoder(encoder_features, x)

        if self.SepActivate:
            x = torch.cat([self.final_conv1(x), self.final_conv2(x)], dim = 1) 
        else:
            x = self.final_conv(x)

        # Apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing accuracy
        #if not self.training:
        #    x = self.final_activation(x)

        return [x]

class Discriminator(nn.Module): # Add sigmoid function as classifier

    def __init__(self, in_channels, out_channels, batch_size = 10, interpolate = True, conv_layer_order = 'crg',
                 init_channel_number = 64):

        super(Discriminator, self).__init__()

        # Number of groups for the GroupNorm
        num_groups = min(init_channel_number // 2, 32)

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, is_max_pool = False, conv_layer_order = conv_layer_order,
                    num_groups = num_groups),
            Encoder(init_channel_number, 2 * init_channel_number, conv_layer_order = conv_layer_order,
                    num_groups = num_groups),
            Encoder(2 * init_channel_number, 4 * init_channel_number, conv_layer_order = conv_layer_order,
                    num_groups = num_groups),
            Encoder(4 * init_channel_number, 8 * init_channel_number, conv_layer_order = conv_layer_order,
                    num_groups = num_groups)
        ])

        self.decoders = nn.ModuleList([
            Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number, interpolate,
                    conv_layer_order = conv_layer_order, num_groups = num_groups),
            Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number, interpolate,
                    conv_layer_order = conv_layer_order, num_groups = num_groups),
            Decoder(init_channel_number + 2 * init_channel_number, init_channel_number, interpolate,
                    conv_layer_order = conv_layer_order, num_groups = num_groups)
        ])

        self.batch_size = batch_size
        # In the last layer a 1×1×1 convolution reduces # of output channels to # of labels
        # self.final_conv = nn.Conv3d(init_channel_number, out_channels, 1)
        #self.final_layer = nn.Linear(init_channel_number, 1)

        self.final_activation = nn.Linear(524288, 1) # TODO
        self.classifier       = nn.Sigmoid()


    def forward(self, x):

        # Encoder part

        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # Reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # Remove the last encoder's output from the list
        # Remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # Decoder part

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)


        #x = self.final_conv(x)
        x = x.view(self.batch_size, -1)
        x = self.final_activation(x)
        x = self.classifier(x)

        # Apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing accuracy
        #if not self.training:
        #    x = self.final_activation(x)

        return x


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d)
    with the number of output channels 'out_channels // 2' and 'out_channels' respectively.
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be change however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ReLU use order='cbr'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.
    Args:
        in_channels (in t): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size ofthe convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size = 3, order='crg', num_groups=32, active_unit='relu'):

        super(DoubleConv, self).__init__()

        if active_unit == 'relu':
            #self.active_unit = nn.ReLU(inplace=True)
            self.active_unit = nn.ReLU()
        elif active_unit == 'elu':
            #self.active_unit = nn.ELU(inplace=True)
            self.active_unit = nn.ELU()
        elif active_unit == 'leaky_relu':
            #self.active_unit = nn.LeakyReLU(inplace=True)
            self.active_unit = nn.LeakyReLU()
        else:
            raise ValueError

        if in_channels < out_channels:
            # If in_channels < out_channels we're in the encoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels // 2
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # Otherwise we're in the decoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self._add_conv(1, conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups)
        # conv2
        self._add_conv(2, conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups)

    def _add_conv(self, pos, in_channels, out_channels, kernel_size, order, num_groups):
        """Add the conv layer with non-linearity and optional batchnorm

        Args:
            pos (int): the order (position) of the layer. MUST be 1 or 2
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            order (string): order of things, e.g.
                'cr' -> conv + ReLU
                'crg' -> conv + ReLU + groupnorm
            num_groups (int): number of groups for the GroupNorm
        """
        assert pos in [1, 2], 'pos MUST be either 1 or 2'
        assert 'c' in order, "'c' (conv layer) MUST be present"
        assert 'r' in order, "'r' (ReLU layer) MUST be present"
        assert order[0] is not 'r', 'ReLU cannot be the first operation in the layer'

        for i, char in enumerate(order):
            if char == 'r':
                self.add_module(f'relu{pos}', self.active_unit)
            elif char == 'c':
                self.add_module(f'conv{pos}', nn.Conv3d(in_channels,
                                                        out_channels,
                                                        kernel_size,
                                                        padding = 1))
            elif char == 'g':
                is_before_conv = i < order.index('c')
                assert not is_before_conv, 'GroupNorm3d MUST go after the Conv3d'
                self.add_module(f'norm{pos}', groupnorm.GroupNorm3d(out_channels, num_groups = num_groups))
            elif char == 'b':
                is_before_conv = i < order.index('c')
                if is_before_conv:
                    self.add_module(f'norm{pos}', nn.BatchNorm3d(in_channels))
                else:
                    self.add_module(f'norm{pos}', nn.BatchNorm3d(out_channels))
            else:
                raise ValueError(f"Unsupported layer type '{char}'. MUST be one of 'b', 'r', 'c'")


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        is_max_pool (bool): if True use MaxPool3d before DoubleConv
        max_pool_kernel_size (tuple): the size of the window to take a max over
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size = 3, is_max_pool = True,
                 max_pool_kernel_size = (2, 2, 2), conv_layer_order = 'crg', num_groups = 32, active_unit = 'relu'):

        super(Encoder, self).__init__()

        self.max_pool = nn.MaxPool3d(kernel_size = max_pool_kernel_size, padding = 1) if is_max_pool else None
        self.double_conv = DoubleConv(in_channels, out_channels,
                                      kernel_size = conv_kernel_size,
                                      order = conv_layer_order,
                                      num_groups = num_groups,
                                      active_unit = active_unit)

    def forward(self, x):

        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.double_conv(x)

        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        interpolate (bool): if True use nn.Upsample for upsampling, otherwise
            learn ConvTranspose3d if you have enough GPU memory and ain't
            afraid of overfitting
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, interpolate, kernel_size = 3,
                 scale_factor = (2, 2, 2), conv_layer_order = 'crg', num_groups = 32, active_unit = 'relu'):

        super(Decoder, self).__init__()

        if interpolate:
            self.upsample = None
        else:
            # make sure that the output size reverses the MaxPool3d TODO: change with different patch
            # D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0]
            self.upsample = nn.ConvTranspose3d(2 * out_channels,
                                               2 * out_channels,
                                               kernel_size = kernel_size,
                                               stride = scale_factor,
                                               padding = 1,
                                               output_padding = 0)

        self.double_conv = DoubleConv(in_channels, out_channels,
                                      kernel_size = kernel_size,
                                      order = conv_layer_order,
                                      num_groups = num_groups,
                                      active_unit = active_unit)

    def forward(self, encoder_features, x):

        if self.upsample is None:
            output_size = encoder_features.size()[2:]
            x = F.interpolate(x, size = output_size, mode = 'nearest')
        else:
            x = self.upsample(x)
        # concatenate encoder_features (encoder path) with the upsampled input across channel dimension
        x = torch.cat((encoder_features, x), dim = 1)
        x = self.double_conv(x)

        return x
