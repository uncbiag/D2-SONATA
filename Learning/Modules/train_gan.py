import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#from torch.optim.lr_scheduler import StepLR  # TODO: may add step scheduler later on

import paths
from losses import GDL_loss
from model.unet import UNet3D, Discriminator
from dataset import ISLESDataset
from trainer_gan import GANModelTrainer
from config import parse_train_GAN_config
from utils import get_logger, get_n_learnable_parameters


def datestr():
    now = time.localtime()
    return '{:04}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
print(datestr())


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger = get_logger("UNet3DTrainer")
    config = parse_train_GAN_config()
    logger.info(config)
    
    ## Nets
    netD = Discriminator(config.in_channels, config.out_channels,
                   batch_size = config.batch_size,
                   init_channel_number = config.init_channel_number,
                   conv_layer_order = config.layer_order,
                   interpolate = config.interpolate)
    netD = netD.to(device)

    netG = UNet3D(config.in_channels, config.out_channels,
                   init_channel_number = config.init_channel_number,
                   conv_layer_order = config.layer_order,
                   interpolate = config.interpolate)
    netG = netG.to(device)
    nets = {"netD": netD, "netG": netG}
    
    ## Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr = config.lr_D) 
    optimizerG = optim.Adam(netG.parameters(), lr = config.lr_G, weight_decay = config.weight_decay)
    optimizers = {"optimizerD": optimizerD, "optimizerG": optimizerG}

    ## Criterions
    criterionGAN = nn.BCELoss()
    criterionGAN = criterionGAN.to(device)
    # Create generator's base loss criterion & Decide where to store validation results
    if config.loss_type == 'L1':
        print('Use L1 Loss for generator base loss')
        fld = 4
        criterionG_base = nn.L1Loss()
    elif config.loss_type == 'L2':
        print('Use L2 Loss for generator base loss')
        fld = 5
        criterionG_base = nn.L2Loss()
    criterionG_base.to(device)
    

    if config.use_gdl:
        fld += 1
    test_save_dir  = os.path.join(paths.TestDirList[fld], config.target_module)
    checkpoint_dir = os.path.join(paths.CheckpointDir, config.target_module)
    print('Testing results will be saved in:', test_save_dir)

    criterionG_GDL = GDL_loss(patch_size = config.RandomCrop, gdl_weight = config.gdl_weight, alpha = config.alpha)
    criterionG_GDL.to(device)
    criterions = {"criterionGAN": criterionGAN, "criterionG_base": criterionG_base, "criterionG_GDL": criterionG_GDL}


    #logger.info(f'Number of learnable params {get_n_learnable_parameters(model)}')
    
    ## Datasets and dataloaders
    train_set    = ISLESDataset(paths.ProcessedTrainingFolder, TargetModule = config.target_module, RandomCrop = config.RandomCrop)
    train_loader = DataLoader(train_set, batch_size = config.batch_size, shuffle = True)
    val_set      = ISLESDataset(paths.ProcessedValidationFolder, TargetModule = config.target_module, RandomCrop = config.RandomCrop)
    val_loader   = DataLoader(val_set, batch_size = config.batch_size, shuffle = True)
    test_set     = ISLESDataset(paths.ProcessedValidationFolder, TargetModule = config.target_module)
    test_loader  = DataLoader(test_set, batch_size = 1, shuffle = False)
    loaders      = {"train": train_loader, "val": val_loader, "test": test_loader} 


    if config.resume: 
        trainer = GANModelTrainer.from_checkpoint(config.resume, model,
                                                optimizer, loss_criterion, gdl_criterion, config.use_gdl, 
                                                loaders,
                                                logger = logger)
    else:
        trainer = GANModelTrainer(nets, optimizers, criterions, config.use_gdl, 
                                device, loaders, checkpoint_dir, 
                                batch_size = config.batch_size,
                                n_init_channel = config.init_channel_number,
                                test_save = test_save_dir,
                                patch_size = config.RandomCrop,
                                print_freq = config.print_freq,
                                max_num_epochs = config.epochs,
                                max_num_iterations = config.iters,
                                max_patience = config.patience, 
                                validate_after_iters = config.validate_after_iters,
                                test_after_iters = config.test_after_iters,
                                log_after_iters = config.log_after_iters,
                                logger = logger)

    trainer.fit()

########################################################################################################################

if __name__ == '__main__':
    main()

