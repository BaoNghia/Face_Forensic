import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from data_loader import transforms as mytransforms
from utils.general import yaml_loader, get_attr_by_name, model_loader
from autoattack.autoattack import AutoAttack
from data_loader.cifar_dataloader import cifar10_dataset, cifar100_dataset


def _pgd_whitebox(model, X, y, adversary):
    features, x4s, logits = model(X)
    out = logits.max(dim=1)[1]
    err = (out != y.data).float().sum()

    X_adv, y_adv = adversary.run_standard_evaluation(X, y, bs=X.size(0), return_labels=True)
    err_adv = (y_adv != y.data).float().sum()
    return err, err_adv


def eval_adv_test(model, device, test_loader, adverary):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    with tqdm(enumerate(test_loader), total = len(test_loader)) as pbar:
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            # pgd attack
            X, y = Variable(inputs, requires_grad=True), Variable(targets)
            err_natural, err_robust = _pgd_whitebox(model, X, y, adverary)
            robust_err_total += err_robust
            natural_err_total += err_natural
            pbar.set_postfix(nat = 1 - (err_natural/y.size(0)).cpu().detach().numpy(), \
                robust = 1 - (err_robust/y.size(0)).cpu().detach().numpy())
    if os.path.exists(log_file):
        open(log_file).write("robust_err_total: " + str(robust_err_total)+ "\n")
        open(log_file).write("natural_err_total: " + str(natural_err_total)+ "\n")
    else:
        with open(log_file, 'w') as f:
            f.write("robust_err_total: " + str(robust_err_total)+ "\n")
            f.write("natural_err_total: " + str(natural_err_total)+ "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-cfg', '--configure', default='cfgs/tense_cifar10_multi.yaml', help='YAML file')
    parser.add_argument('-ckpt', '--checkpoint', default=None, help = 'checkpoint path')
    parser.add_argument('--version', type=str, default='standard')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    checkpoint_path = 'logs/CIFAR10_multi/2022-05-16-09h17/robust_last.ckpt'
    log_file = os.path.join("/".join(checkpoint_path.split("/")[:-1]), "eval_log.txt")
    print("Testing process beginning here....")
    
    # read configure file
    cfg = yaml_loader(args.configure)
    # load model
    print("Loading model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    all_model = model_loader(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    test_model = all_model['model_robust']
    try:
        test_model.load_state_dict(checkpoint['state_dict'])
    except:
        test_model = nn.DataParallel(test_model).to(device)
        test_model.load_state_dict(checkpoint['state_dict'])
    test_model = test_model.to(device)
    test_model.eval()

    adversarial_cfg = cfg.get("adversarial")
    norm = 'Linf' if adversarial_cfg['norm'] == "np.inf" else f"L{adversarial_cfg['norm']}"
    epsilon = adversarial_cfg['epsilon']
    adversary = AutoAttack(
        test_model, 
        norm = norm, 
        eps = epsilon,
        version = args.version,
        device = device, 
        log_path = log_file,
        verbose = True, 
    )
    adversary.seed = 0
    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2


    # Create dataset
    batch_size = 64
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    trainset, testset = cifar10_dataset(cfg)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    eval_adv_test(test_model, device, test_loader, adversary)
