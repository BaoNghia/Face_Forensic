import os, time, logging
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from models.PGD import Adversarial
from utils import callbacks, metrics_loader, general
from data_loader.dataloader import get_dataset
from utils.general import (model_loader,  get_optimizer, get_loss_fn,\
    get_lr_scheduler, yaml_loader, save_best_checkpoint, save_last_checkpoint)
import argparse
import tester, trainer


def get_dataloader():
    # setup data loader
    from torchvision import datasets, transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    batch_size = 512
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    trainset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    testset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

# from torchsampler import ImbalancedDatasetSampler
def main(cfg, all_model, log_dir, checkpoint=None):            
    if checkpoint is not None:
        print("...Load checkpoint from {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        all_model = {name: model.load_state_dict(checkpoint['state_dict']) for name, model in all_model.items()}
        print("...Checkpoint loaded")

    # Checking cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: {} ".format(device))

    # Convert to suitable device
    all_model = {name: nn.DataParallel(model).to(device) for name, model in all_model.items()}
    num_parameter = {name: sum(p.numel() for p in model.parameters()) for name, model in all_model.items()}
    logging.info(f"Number parameters of model: {num_parameter}")

    # using parsed configurations to create a dataset
    # Create dataset
    num_of_class = len(cfg["data"]["label_dict"])
    train_set, valid_set, test_set = get_dataset(cfg)
    # Dataloader
    batch_size = int(cfg["data"]["batch_size"])
    train_loader = torch.utils.data.DataLoader(
		train_set, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
		valid_set, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
		test_set, batch_size=32, shuffle=False
    )
    print("Dataset and Dataloaders created")

    # create a metric for evaluating
    metric_names = cfg["train"]["metrics"]
    train_metrics = metrics_loader.Metrics(metric_names)
    valid_metrics = metrics_loader.Metrics(metric_names)
    print("Metrics implemented successfully")

    ## read settings from json file
    ## initlize optimizer from config
    ## optimizer
    optimizer_module, optimizer_params = get_optimizer(cfg)
    parameters = [{"params": model.parameters()} for name, model in all_model.items()]
    optimizer = optimizer_module(parameters, **optimizer_params)
    ## initlize sheduler from config
    scheduler_module, scheduler_params = get_lr_scheduler(cfg)
    scheduler = scheduler_module(optimizer, **scheduler_params)
    # scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
    loss_fn, loss_params = get_loss_fn(cfg)
    criterion = loss_fn(**loss_params)
    # Create Adversarial_model
    model_robust = all_model["model_robust"]
    model_natural = all_model["model_natural"]

    print("\nTraing shape: {} samples".format(len(train_loader.dataset)))
    print("Validation shape: {} samples".format(len(valid_loader.dataset)))
    print("Beginning training...")

    # initialize the early_stopping object
    save_mode = cfg["train"]["mode"]
    early_patience = cfg["train"]["patience"]
    checkpoint_path = os.path.join(log_dir, "best.ckpt")
    early_stopping = callbacks.EarlyStopping(patience=early_patience,
                                            mode = save_mode, 
                                            delta = 0,
                                            path = checkpoint_path)
    
    # training models
    train_loader, valid_loader = get_dataloader()
    logging.info("--"*50)
    num_epochs = int(cfg["train"]["num_epochs"])
    t0 = time.time()
    best_valid_lost = np.inf

    for epoch in range(num_epochs):
        t1 = time.time()
        print(('\n' + '%13s' * 3) % ('Epoch', 'gpu_mem', 'mean_loss'))
        train_loss, train_acc, train_result = trainer.train_epoch(epoch, num_epochs, device, 
                                                                model_robust, model_natural,
                                                                train_loader, train_metrics,
                                                                criterion, optimizer, cfg
        )
        valid_loss, valid_acc, valid_result = trainer.valid_epoch(device, model_robust, model_natural,
                                                                valid_loader, valid_metrics, criterion,
                                                                cfg, train_loss, train_acc,
        )
        scheduler.step(valid_loss)


        print("Valid result: ", valid_result)
        ## log to file 
        logging.info("\n------Epoch {} / {}, Training time: {:.4f} seconds------"\
            .format(epoch, num_epochs, (time.time() - t1)))
        logging.info(f"Training loss: {train_loss} \n Training metrics: {train_result}")
        logging.info(f"Validation loss: {valid_loss} \n Validation metrics: {valid_result}")
        
        ## tensorboard writer
        tb_writer.add_scalar("Training Loss", train_loss, epoch)
        tb_writer.add_scalar("Valid Loss", valid_loss, epoch)
        for metric_name in metric_names:
            tb_writer.add_scalar(f"Training {metric_name}", train_result[metric_name], epoch)
            tb_writer.add_scalar(f"Validation {metric_name}", valid_result[metric_name], epoch)
        
        # Save model
        robust_checkpoint = {
            'epoch': epoch,
            'valid_loss': valid_loss,
            'model': model_robust,
            'state_dict': model_robust.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_last_checkpoint(robust_checkpoint, log_dir, name = "robust")
        if valid_loss < best_valid_lost:
            best_valid_lost = valid_loss
            save_best_checkpoint(robust_checkpoint, log_dir, name = "robust")
        

        # if save_mode == "min":
        #     early_stopping(valid_loss, train_checkpoint)
        # else:
        #     early_stopping(valid_acc, train_checkpoint)
        # if early_stopping.early_stop:
        #     logging.info("Early Stopping!!!")
        #     break

    # # testing on test set
    # # load the test model and making inference
    # print("\n==============Inference on the testing set==============")
    # best_checkpoint = torch.load(checkpoint_path)
    # test_model = best_checkpoint['model']
    # test_model.load_state_dict(best_checkpoint['state_dict'])
    # test_model = test_model.to(device)
    # test_model.eval()

    # # logging report
    test_model = model_robust.to(device)
    test_model.eval()
    # report = tester.test_result(test_model, test_loader, device, cfg)
    report = tester.test_result(test_model, valid_loader, device, cfg)
    logging.info(f"\nClassification Report: \n {report}")
    logging.info("Completed in {:.3f} seconds. ".format(time.time() - t0))

    print(f"Classification Report: \n {report}")
    print("Completed in {:.3f} seconds.".format(time.time() - t0))
    print(f"-------- Checkpoints and logs are saved in ``{log_dir}`` --------")
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-c', '--configure', default='cfgs/tense.yaml', help='YAML file')
    parser.add_argument('-cp', '--checkpoint', default=None, help = 'checkpoint path for transfer learning')
    args = parser.parse_args()
    checkpoint = args.checkpoint

    # read configure file
    config = yaml_loader(args.configure) # config dict
    ## comment for this experiment: leave here
    comment = config["session"]["_comment_"]

    ## create dir to save log and checkpoint
    save_path = config['session']['save_path']
    time_str = str(datetime.now().strftime("%Y-%m-%d-%Hh%M"))
    project_name = config["session"]["project_name"]
    log_dir = os.path.join(save_path, project_name, time_str)

    ## create logger
    tb_writer = general.make_writer(log_dir = log_dir)
    text_logger = general.log_initilize(log_dir)
    print(f"Start Tensorboard with tensorboard --logdir {log_dir}, view at http://localhost:6006/")
    logging.info(f"Start Tensorboard with tensorboard --logdir {log_dir}, view at http://localhost:6006/")
    logging.info(f"Project name: {project_name}")
    logging.info(f"CONFIGS: \n {config}")

    ## Create model
    all_model = model_loader(config)
    print("Create model Successfully !!!")
    num_parameter = {name: sum(p.numel() for p in model.parameters()) for name, model in all_model.items()}
    print(f"Number parameters of model: {num_parameter}")
    # time.sleep(1.8)

    best_ckpt = main(
        cfg = config,
        all_model = all_model,
        log_dir = log_dir,
        checkpoint = checkpoint,
    )
