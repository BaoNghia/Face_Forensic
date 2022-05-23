import os, re, yaml
import math
import logging
import torch
import importlib
import numpy as np
import random
from torch.autograd import Variable
from utils import losses as custom_loss

class i_class_idx(object):
    def __init__(self, targets, num_class):
        self.targets = targets
        self.num_class = num_class

    def get_idx(self):
        self.i_idx = []
        for j in range(self.num_class):
            idx = np.where(np.array(self.targets) == j)[0]
            random.shuffle(idx)
            self.i_idx.append(idx)

    def get_batch(self, batch_targets, delete = False):
        x1_pos_idx = np.zeros(len(batch_targets), dtype = np.int)
        for j in range(len(batch_targets)):
            target = batch_targets[j].item()
            to_batch = np.random.choice(self.i_idx[target], 1, replace = False)
            x1_pos_idx[j] = to_batch
            if delete:
                self.i_idx[target] = np.setdiff1d(self.i_idx[target], to_batch, assume_unique = True)
        return x1_pos_idx

def to_variable(dataset, idx_list):
    for i, idx in enumerate(idx_list):
        if i == 0:
            input_batch = dataset[idx][0]
            input_batch.unsqueeze_(0)
            target_batch = torch.tensor(dataset[idx][1])
            target_batch.unsqueeze_(0)
        else:
            one_instance = dataset[idx][0]
            one_instance.unsqueeze_(0)
            input_batch = torch.cat([input_batch, one_instance], 0)
            one_target = torch.tensor(dataset[idx][1])
            one_target.unsqueeze_(0)
            target_batch = torch.cat([target_batch, one_target], 0)
    input_batch = Variable(input_batch)
    target_batch = Variable(target_batch)
    return input_batch, target_batch

def most_confussion(logits, targets):
    a = logits.detach().clone()
    a[np.arange(targets.shape[0]),targets] = -1e4
    _, labels = a.max(1)
    return labels


def get_attr_by_name(func_str):
    """
    Load function by full name
    :param func_str:
    :return: fn, mod
    """
    module_name, func_name = func_str.rsplit('.', 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func, module, func_name

def model_loader(config):
    all_model = {}
    for key, model_dict in config.items():
        if 'model' in key.lower():
            func, _, _ = get_attr_by_name(model_dict['model.class'])
            del model_dict['model.class']
            all_model[key] = func(**model_dict)
    return all_model

def get_optimizer(config):
    cfg =  config.get("optimizer")
    optimizer_name = cfg["name"]
    try:
        optimizer = getattr(torch.optim, optimizer_name,\
            "The optimizer {} is not available".format(optimizer_name))
    except:
        optimizer = getattr(torch.optim, optimizer_name,\
            "The optimizer {} is not available".format(optimizer_name))
    del cfg['name']
    return optimizer, cfg


def adjust_learning_rate(optimizer, epoch, init_lr):
    """decrease the learning rate"""
    lr = init_lr
    if epoch==0:
       lr=0.02
    if epoch >= 76:
        lr = init_lr * 0.1
    if epoch >= 91:
        lr = init_lr * 0.01
    if epoch >= 101:
        lr = init_lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr_scheduler(config):
    cfg = config.get("scheduler")
    scheduler_name = cfg["name"]
    try:
        # if the lr_scheduler comes from torch.optim.lr_scheduler package
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)
    except:
        # use custom loss
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_name,\
            "The scheduler {} is not available".format(scheduler_name))
    del cfg['name']
    return scheduler, cfg

def get_loss_fn(config):
    loss_dict = config.get("loss")
    try:
        # if the loss function comes from nn package
        criterion = getattr(torch.nn, loss_dict['name'])
    except:
        # use custom loss
        criterion = getattr(custom_loss, loss_dict['name'],
            "The loss {} is not available".format(loss_dict['name']))
    del loss_dict['name']
    return criterion, loss_dict

def make_dir_epoch_time(base_path, session_name, time_str):
    """
    make a new dir on base_path with epoch_time
    :param base_path:
    :return:
    """
    new_path = os.path.join(base_path, session_name + "_" + time_str)
    os.makedirs(new_path, exist_ok=True)
    return new_path

def save_last_checkpoint(checkpoint, log_dir, name):
    cp_path = os.path.join(log_dir, f"{name}_last.ckpt")
    torch.save(checkpoint, cp_path)
    return cp_path

def save_best_checkpoint(checkpoint, log_dir, name):
    cp_path = os.path.join(log_dir, f"{name}_best.ckpt")
    torch.save(checkpoint, cp_path)
    return cp_path

def yaml_loader(yaml_file):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.')
    )
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=loader) # cfg dict
    return config

def log_initilize(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create error file handler and set level to error
    log_file = os.path.join(log_dir, "model_logs.txt")
    handler = logging.FileHandler(log_file, "a", encoding=None, delay="true")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.terminator = "\n"
    # add handler to logger
    logger.addHandler(handler)
    return logger

def make_writer(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def to_PILImage(x, index = 0):
    from torchvision import transforms as T
    trans = T.ToPILImage()
    img = trans(x[index])
    return img
