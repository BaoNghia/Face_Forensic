import os
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn

from utils.attacks import Attacks
from utils import callbacks, metrics_loader
from utils.general import (
    make_writer, log_initilize, yaml_loader,
    model_loader, get_optimizer, get_loss_fn,
    adjust_learning_rate, get_lr_scheduler,
    save_best_checkpoint, save_last_checkpoint)

import tester, trainer
from data_loader.dataloader import get_dataset, get_dataloader
from data_loader.cifar_dataloader import cifar10_dataloader, cifar100_dataloader

# from torchsampler import ImbalancedDatasetSampler
def main(cfg, model_teacher, log_dir):
    # Checking cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: {} ".format(device))

    # Convert to suitable device
    if device.type == 'cpu':
        model_teacher = model_teacher.to(device)
    else:
        model_teacher = nn.DataParallel(model_teacher).to(device)

    # create a metric for evaluating
    metric_names = cfg["train"]["metrics"]
    train_metrics = metrics_loader.Metrics(metric_names)
    valid_metrics = metrics_loader.Metrics(metric_names)
    print("Metrics implemented successfully")

    ## read settings from json file
    ## initlize optimizer from config
    ## optimizer
    optimizer_module, optimizer_params = get_optimizer(cfg)
    optimizer = optimizer_module(model_teacher.parameters(), **optimizer_params)
    ## initlize sheduler from config
    scheduler_module, scheduler_params = get_lr_scheduler(cfg)
    scheduler = scheduler_module(optimizer, **scheduler_params)
    ## get Loss function
    loss_fn, loss_params = get_loss_fn(cfg)
    criterion = loss_fn(**loss_params)

    # using parsed configurations to create a dataset
    # Create dataset
    train_data, valid_data, test_data = get_dataset(cfg)
    batch_size = int(cfg["data"]["batch_size"])
    train_loader, valid_loader, test_loader = get_dataloader(train_data, valid_data, test_data, batch_size)

    print("Dataset and Dataloaders created")
    print("\nTraing shape: {} samples".format(len(train_loader.dataset)))
    print("Validation shape: {} samples".format(len(valid_loader.dataset)))
    print("Beginning training...")

    # training models
    logging.info("--"*50)
    num_epochs = int(cfg["train"]["num_epochs"])
    t0 = time.time()
    best_valid_lost = np.inf

    for epoch in range(num_epochs):
        t1 = time.time()
        print(('\n' + '%13s' * 4) % ('Epoch', 'gpu_mem', 'mean_loss', 'mean_acc'))
        train_loss, train_acc, train_result = trainer.train_teacher_epoch( \
            epoch, num_epochs, device, model_teacher,\
            train_loader, train_metrics,\
            criterion, optimizer, cfg\
        )

        valid_loss, valid_acc, valid_result = trainer.valid_teacher_epoch( \
            device, model_teacher,\
            valid_loader, valid_metrics,\
            criterion, train_loss, train_acc, cfg\
        )
        scheduler.step(valid_loss)

        print("Train result: ", train_result)
        print("Valid result: ", valid_result)
        ## log to file 
        logging.info("\n------Epoch {} / {}, Training time: {:.4f} seconds------"\
            .format(epoch, num_epochs, (time.time() - t1)))
        logging.info(f"Training loss: {train_loss} \nTraining metrics: {train_result}")
        logging.info(f"Validation loss: {valid_loss} \nValidation metrics: {valid_result}")
        
        ## tensorboard writer
        tb_writer.add_scalar("Training Loss", train_loss, epoch)
        tb_writer.add_scalar("Valid Loss", valid_loss, epoch)
        for metric_name in metric_names:
            tb_writer.add_scalar(f"Training {metric_name}", train_result[metric_name], epoch)
            tb_writer.add_scalar(f"Validation {metric_name}", valid_result[metric_name], epoch)
        
        # Save model
        model_checkpoint = {
            'epoch': epoch,
            'valid_loss': valid_loss,
            'model': model_teacher,
            'state_dict': model_teacher.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        last_cpkt_path = save_last_checkpoint(model_checkpoint, log_dir, name = "")
        if valid_loss < best_valid_lost:
            best_valid_lost = valid_loss
            best_cpkt_path = save_best_checkpoint(model_checkpoint, log_dir, name = "")

    ## logging report
    test_model = model_teacher.to(device)
    test_model.eval()
    report = tester.test_result(test_model, test_loader, device, label_name = cfg.get("data")["label_dict"])
    # report = tester.test_result(test_model, valid_loader, device, cfg.get("data")["label_dict"])
    logging.info(f"\nClassification Report: \n {report}")
    logging.info("Completed in {:.3f} seconds. ".format(time.time() - t0))

    print(f"Classification Report: \n {report}")
    print("Completed in {:.3f} seconds.".format(time.time() - t0))
    print(f"-------- Checkpoints and logs are saved in ``{log_dir}`` --------")
    return best_cpkt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-cfg', '--configure', default='cfgs/tense_teacher_fl.yaml', help='YAML file')
    parser.add_argument('-ckpt', '--pretrained', default=None, help = 'checkpoint path for transfer learning')
    args = parser.parse_args()
    pretrained = args.pretrained

    # read configure file
    config = yaml_loader(args.configure) # config dict
    ## comment for this experiment: leave here
    comment = config["session"]["_comment_"]

    ## create dir to save log and checkpoint
    save_path = config['session']['save_path']
    time_str = str(time.strftime("%Y-%m-%d-%Hh%M", time.localtime()))
    project_name = config["session"]["project_name"]
    log_dir = os.path.join(save_path, project_name, f'{time_str}-{comment}')

    ## create logger
    tb_writer = make_writer(log_dir = log_dir)
    text_logger = log_initilize(log_dir)
    print(f"Start Tensorboard with tensorboard --logdir {log_dir}, view at http://localhost:6006/")
    logging.info(f"Start Tensorboard with tensorboard --logdir {log_dir}, view at http://localhost:6006/")
    logging.info(f"Project name: {project_name}")
    logging.info(f"CONFIGS: \n {config}")

    ## Create model
    all_model = model_loader(config)
    model = all_model['model_teacher']
    if pretrained is not None:
        print("...Load Pretrain from {}".format(pretrained))
        pretrained = torch.load(pretrained)
        model.load_state_dict(pretrained['state_dict'])
        print("...Pretrain is loaded")
    else:
        print("Train from scratch")
    print("Create model Successfully !!!")
    num_parameter = sum(p.numel() for p in model.parameters())
    print(f"Number parameters of model: {num_parameter}")
    logging.info(f"Number parameters of model: {num_parameter}")

    best_ckpt_path = main(
        cfg = config,
        model_teacher = model,
        log_dir = log_dir,
    )