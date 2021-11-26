import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from data_loader import transforms as mytransforms
from utils.general import yaml_loader, get_attr_by_name
from autoattack import AutoAttack


def _pgd_whitebox(model, X, y, adversary):
    out = model(X).max(dim=1)[1]
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

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, adverary)
        robust_err_total += err_robust
        natural_err_total += err_natural

    open(log_file).write("robust_err_total: " + str(robust_err_total)+ "\n")
    open(log_file).write("natural_err_total: " + str(natural_err_total)+ "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-cfg', '--configure', default='cfgs/tenes.yaml', help='YAML file')
    parser.add_argument('-cp', '--checkpoint', default=None, help = 'checkpoint path')
    parser.add_argument('--version', type=str, default='standard')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    log_file = os.path.join("/".join(checkpoint_path.split("/")[:-1]), "eval_log.txt")
    print("Testing process beginning here....")
    
    # read configure file
    cfg = yaml_loader(args.configure)
    # load model
    print("Loading model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path)
    test_model = checkpoint['model']
    test_model.load_state_dict(checkpoint['state_dict'])
    test_model = test_model.to(device)
    test_model.eval()

    adversarial_cfg = cfg.get("adversarial")
    norm = 'Linf' if adversarial_cfg['norm'] == "np.inf" else f"L{adversarial_cfg['norm']}"
    epsilon = adversarial_cfg['epsilon']
    adversary = AutoAttack(test_model, 
                            norm=norm, 
                            eps= epsilon,
                            version= args.version,
                            log_path = log_file)
    adversary.seed = 0
    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2

    test_csv = cfg["data"]["test_csv_name"]
    test_set = pd.read_csv(test_csv)
    # Get Custom Dataset inherit from torch.utils.data.Dataset
    dataset, _, _ = get_attr_by_name(cfg['data']['data.class'])
    test_set = dataset(test_set, transform = mytransforms.test_transform)
    test_loader = data.DataLoader(test_set, batch_size=64, num_workers=0, shuffle=False)
    eval_adv_test(test_model, device, test_loader, adversary)
