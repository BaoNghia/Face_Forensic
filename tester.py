import os
import json
import PIL
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.utils.data as data
from data_loader import transforms as mytransforms
from utils.general import yaml_loader, model_loader, get_attr_by_name

from sklearn.metrics import classification_report



def test_result(model, test_loader, device, label_name):
    # testing the model by turning model "eval" mode
    with torch.no_grad():
        model.eval()
        list_labels = []
        list_preds = []
        
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                _, _, outputs = model(inputs)
                _, preds = torch.max(outputs.data, dim=-1)
                # outputs_softmax = torch.softmax(outputs, dim=-1)
            
            list_labels.extend(targets.cpu().detach().numpy())
            list_preds.extend(preds.cpu().detach().numpy())
    return (classification_report(list_labels, list_preds, target_names=label_name, zero_division = 1))


def test(model, device, test_loader, criterion, test_metrics):
    #validate-the-model
    with tqdm(enumerate(test_loader), total = len(test_loader)) as pbar:
        with torch.no_grad():
            valid_loss = 0
            all_labels = []
            all_preds = []
            model.eval()
            for batch_idx, (inputs, targets) in pbar:
                if not type(inputs) in (tuple, list):
                    inputs = (inputs,)
                # move-tensors-to-GPU
                inputs = tuple(input.to(device) for input in inputs)
                targets = targets.to(device)
                # forward model
                outputs = model(*inputs)
                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)
                # Calculate loss
                loss_inputs = outputs
                if targets is not None:
                    targets = (targets,)
                    loss_inputs += targets
                loss_outputs = criterion(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                # update-validation-loss
                valid_loss += loss.item()
                ## calculate training metrics
                outputs_softmax = torch.softmax(outputs[0], dim=-1)
                    
                probs, preds = torch.max(outputs_softmax.data, dim=-1)
                all_labels.extend(targets[0].cpu().detach().numpy())
                all_preds.extend(preds.cpu().detach().numpy())

        test_metrics.step(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=cfg["data"]["label_dict"])
    return report, test_metrics.last_step_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-cfg', '--configure', default='cfgs/tense_teacher_fl.yaml', help='YAML file')
    parser.add_argument('-ckpt', '--checkpoint', default=None, help = 'checkpoint path')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    print("Testing process beginning here....")
    
    # read configure file
    cfg = yaml_loader(args.configure)

    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    all_model = model_loader(cfg)
    test_model = all_model['model_teacher']
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    test_model.load_state_dict(checkpoint['state_dict'])
    test_model = test_model.to(device)
    test_model.eval()

    print("Inference on the testing set")
    test_csv = cfg["data"]["test_csv_name"]
    test_set = pd.read_csv(test_csv)

    # prepare the dataset
    dataset, _, _ = get_attr_by_name(cfg['data']['data.class'])
    test_set = dataset(test_set, transform = mytransforms.test_transform)
    test_loader = data.DataLoader(test_set, batch_size=32, num_workers=0, shuffle=False)
    print(test_result(test_model, test_loader, device, label_name = cfg.get("data")["label_dict"]))