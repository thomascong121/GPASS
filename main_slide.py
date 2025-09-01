import hydra
from omegaconf import DictConfig, OmegaConf
from data.dataset import *
from network.s3ngan import *
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from utils.util import *
import cv2
import torch
import numpy as np
import random
import logging
import os
from datetime import datetime
from torchvision.transforms.functional import to_pil_image
from utils.WholeSlideImage import WholeSlideImage

# Configure logger
def setup_logger(name, log_level=logging.INFO):
    """Set up a logger with both file and console handlers."""
    # Remove any existing handlers from the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a custom logger
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Prevent the logger from propagating to the root logger
    logger.propagate = False
    logger.setLevel(log_level)
    
    # Create handlers if they don't exist
    if not logger.handlers:
        log_file = f'logs/{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatters
        log_format = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
        
        # Set formatters
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Initialize logger with DEBUG level to capture all messages
logger = setup_logger('slide_processor', log_level=logging.DEBUG)

# Set handler levels to DEBUG to ensure all messages are captured
for handler in logger.handlers:
    handler.setLevel(logging.DEBUG)


@hydra.main(config_path='./configs', config_name='config.yaml')
def main(cfg: DictConfig):

    logger.info("Configuration loaded successfully")
    logger.debug(f"Configuration details:\n{OmegaConf.to_yaml(cfg)}")

    source_wsi_dataset_train = WSISlideDatasetTT(split='source',
                                                 data_path=cfg.dataset.opt_dataset['data_root'],
                                                 test=False)
    target_wsi_dataset_train = WSISlideDatasetTT(split='target',
                                                 data_path=cfg.dataset.opt_dataset['data_root'])
    source_wsi_dataset_test = WSISlideDatasetTT(split='source',
                                                data_path=cfg.dataset.opt_dataset['data_root'],
                                                use_all=True,
                                                test=True)

    logger.info(f'Source dataset size: {len(source_wsi_dataset_train)}')
    logger.info(f'Target dataset size: {len(target_wsi_dataset_train)}')
    logger.info(f'Test dataset size: {len(source_wsi_dataset_test)}')

    model = twodecoderGANModel(cfg)
    overall_best_psnr = 0
    overall_best_ssim = 0
    overall_best_epoch = 0
    start_epoch = cfg.run.opt_run['which_epoch'] + 1 if cfg.run.opt_run['continue_train'] else cfg.run.opt_run['which_epoch']
    if cfg.run.opt_run['stage'] == 'test':
        # fix seed in test
        seed = 42  # Using a fixed seed value
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        overall_psnr, overall_ssim = [], []
        overall_nmi_base = []
        overall_nmi_pred = []       
        logger.info('Starting testing phase...')
        valid_slide = 0
        if cfg.dataset.opt_dataset['data_root'].split('/')[-1] == 'SN_TT':
            source_wsi_dataset_test = source_wsi_dataset_train
        for i in range(len(source_wsi_dataset_test)):
            logger.info(f'Processing slide {i+1}/{len(source_wsi_dataset_test)}')
            source_slide, source_slide_coord = source_wsi_dataset_test[i]
            source_slide_name = source_slide.split('/')[-1].split('.')[0]
            test_out_dir = f'%s/%s/normalised/%d/test/%s/' % (str(cfg.run.opt_run['checkpoints_dir']),
                                                             cfg.dataset.opt_dataset['name'],
                                                             cfg.run.opt_run['which_epoch'],
                                                             source_slide_name)
            if os.path.exists(test_out_dir) and cfg.run.opt_run['auto_skip']:
                logger.info(f'Skipping slide {source_slide_name} - already processed')
                continue
            source_slide_openslide = openslide.open_slide(source_slide)
            source_slide_dataset = Whole_Slide_Bag_FP(file_path=source_slide_coord,
                                                      wsi=source_slide_openslide,
                                                      img_transforms=None)
            test_dataset = SinglePatchDataset(source_slide_dataset, cfg.dataset.opt_dataset)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=cfg.run.opt_run['batchSize'],
                shuffle=False,
                num_workers=8,
                drop_last=True)
            try:
                slide_psnr, slide_ssim, slide_nmi_base, slide_nmi_pred = model.test(test_loader, 
                                                                                stage='test', 
                                                                                save=False, 
                                                                                source_slide_name=source_slide_name,
                                                                                logger=logger)
            except Exception as e:
                logger.error(f'Error processing slide {source_slide_name}: {e}')
                continue
            overall_psnr.append(slide_psnr.cpu().numpy())
            overall_ssim.append(slide_ssim.cpu().numpy())
            overall_nmi_base.append(slide_nmi_base)
            overall_nmi_pred.append(slide_nmi_pred)
            valid_slide += 1
            # if valid_slide >= 3:
            #     break
        # print the mean and std of psnr and ssIM
        logger.debug(f'NMI base values: {overall_nmi_base}')
        logger.debug(f'NMI pred values: {overall_nmi_pred}')
        
        # Calculate metrics
        psnr_mean, psnr_std = np.mean(overall_psnr), np.std(overall_psnr)
        ssim_mean, ssim_std = np.mean(overall_ssim), np.std(overall_ssim)
        nmi_base_std, nmi_base_mean = np.std(overall_nmi_base), np.mean(overall_nmi_base)
        nmi_pred_std, nmi_pred_mean = np.std(overall_nmi_pred), np.mean(overall_nmi_pred)
        
        # Convert to numpy arrays and save
        nim_base_np = np.asarray(overall_nmi_base)
        nim_pred_np = np.asarray(overall_nmi_pred)
        
        # Define output paths
        output_dir = os.path.join(
            str(cfg.run.opt_run['checkpoints_dir']),
            cfg.dataset.opt_dataset['name'],
            'normalised',
            str(cfg.run.opt_run['which_epoch']),
            'test'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Save NMI data
        np.save(os.path.join(output_dir, 'nim_base.npy'), nim_base_np)
        np.save(os.path.join(output_dir, 'nim_pred.npy'), nim_pred_np)
        
        # Log results
        logger.info('\n' + '='*50)
        logger.info('TEST RESULTS SUMMARY')
        logger.info('='*50)
        logger.info(f'PSNR: Mean = {psnr_mean:.3f} ± {psnr_std:.3f}')
        logger.info(f'SSIM: Mean = {ssim_mean:.3f} ± {ssim_std:.3f}')
        logger.info('-'*50)
        logger.info(f'NMI Base: Mean = {nmi_base_mean:.3f} ± {nmi_base_std:.3f} (CoV = {nmi_base_std/nmi_base_mean:.3f})')
        logger.info(f'NMI Pred: Mean = {nmi_pred_mean:.3f} ± {nmi_pred_std:.3f} (CoV = {nmi_pred_std/nmi_pred_mean:.3f})')
        logger.info('-'*50)
        logger.info(f'Processed {valid_slide} out of {len(source_wsi_dataset_test)} slides')
        logger.info('='*50 + '\n')
        logger.info(f'NMI saved in {output_dir}')
        
    else:
        logger.info('Starting training phase...')
        for epoch in range(start_epoch, cfg.run.opt_run['n_epoch'] + 1):
            epoch_loss = {}
            iters = 0
            idx_all_source = np.random.permutation(len(source_wsi_dataset_train))
            pbar = tqdm(range(len(target_wsi_dataset_train)))
            for i in pbar:
                target_slide, target_slide_coord = target_wsi_dataset_train[i]
                idx = idx_all_source[i % len(idx_all_source)]
                source_slide, source_slide_coord = source_wsi_dataset_train[idx]
                target_slide = openslide.open_slide(target_slide)
                source_slide = openslide.open_slide(source_slide)
                target_slide_dataset = Whole_Slide_Bag_FP(file_path=target_slide_coord,
                                                          wsi=target_slide,
                                                          img_transforms=None)
                source_slide_dataset = Whole_Slide_Bag_FP(file_path=source_slide_coord,
                                                          wsi=source_slide,
                                                          img_transforms=None)
                train_dataset = AlignedPatchDataset(source_slide_dataset,
                                                    target_slide_dataset,
                                                    cfg.dataset.opt_dataset)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=cfg.run.opt_run['batchSize'],
                    shuffle=True,
                    num_workers=8,
                    pin_memory=True,
                    drop_last=True)

                for i, (inputs) in enumerate(train_loader):
                    model.set_input(inputs)
                    model.optimize_parameters()
                    current_error = model.get_current_errors()
                    for loss_name in current_error:
                        if loss_name not in epoch_loss:
                            epoch_loss[loss_name] = current_error[loss_name]
                        else:
                            epoch_loss[loss_name] += current_error[loss_name]
                    iters += 1
                # pbar.set_description(f"{i}/{len(train_loader)}")
                    # break
            output = "===> Epoch {%d} Complete: Avg." % epoch
            for loss_name in epoch_loss:
                output += '%s: %.3f ' % (loss_name, epoch_loss[loss_name] / iters)
            print(output)
            adjust_learning_rate(model.optimizer_D, epoch, cfg.run.opt_run, cfg.run.opt_run['lr_D'])
            adjust_learning_rate(model.optimizer_G, epoch, cfg.run.opt_run, cfg.run.opt_run['lr_G'])

            model.get_current_visuals(epoch)
            model.save(epoch)

if __name__ == '__main__':
    main()