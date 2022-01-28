import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.general import convert_size, to_PILImage
from utils.attacks import generate_TRADES, generate_PGD

def train_epoch(
        epoch, num_epochs, device,
        model_robust, model_teacher,
        train_loader, train_metrics,
        criterion, optimizer, attacker,
        cfg,
    ):
    ## training-the-model
    with tqdm(enumerate(train_loader), total = len(train_loader)) as pbar:
        train_loss = 0
        mloss = 0
        correct = 0
        model_teacher.train()
        for batch_idx, (inputs, targets) in pbar:
            ## move-tensors-to-GPU
            inputs, targets = inputs.to(device), targets.to(device)
            ## generate adversarial_sample
            optimizer.zero_grad()
            # adv_inputs = attacker.perturb_PGD(inputs, targets)
            adv_inputs = generate_TRADES(model_robust, inputs, cfg.get("adversarial"))
            ## zero the gradient beforehand
            model_robust.train()
            optimizer.zero_grad()
            out_adv = model_robust(adv_inputs)
            out_natural = model_robust(inputs)
            out_orig = model_teacher(inputs)
            ## forward model and compute loss
            loss = criterion(out_adv, out_natural, out_orig, targets)
            loss.backward()
            optimizer.step()
            ## update training-loss
            train_loss += loss.item() * inputs.size(0)
            ## calculate training metrics
            outputs = model_robust(inputs)
            _, preds = torch.max(outputs.data, dim=-1)
            correct += torch.sum(preds.data == targets.data).item()
            train_metrics.step(preds.cpu().detach().numpy(), targets.cpu().detach().numpy())

            ## pbar
            mem = convert_size(torch.cuda.memory_reserved()) if torch.cuda.is_available() else "0 GB"  # (GB)
            mloss = (mloss * batch_idx + loss.item())/(batch_idx + 1)
            s = ('%13s' * 2 + '%13.4g' * 1) % ('%g/%g' % (epoch, num_epochs - 1), mem, mloss)
            pbar.set_description(s)
            pbar.set_postfix(lr = optimizer.param_groups[0]['lr'])

        train_loss = train_loss/len(train_loader.dataset)
        train_acc = correct/len(train_loader.dataset)

    return train_loss, train_acc, train_metrics.epoch()
        
    
def valid_epoch(
        device, model_robust, model_teacher,
        valid_loader, valid_metrics, criterion,
        train_loss, train_acc, attacker, cfg
    ):
    #validate-the-model
    criterion_ori = nn.CrossEntropyLoss()
    with tqdm(enumerate(valid_loader), total = len(valid_loader)) as pbar:
        pbar.set_description(('%13s'  + '%13s' * 3) % ('Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'))
        with torch.no_grad():
            correct = 0
            valid_loss = 0
            all_labels = []
            all_preds = []
            model_teacher.eval()
            model_robust.eval()
            for batch_idx, (inputs, targets) in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_robust(inputs)
                loss = criterion_ori(outputs, targets)
                valid_loss += loss.item() * inputs.size(0)
                ## calculate training metrics
                _, preds = torch.max(outputs.data, dim=-1)
                correct += torch.sum(preds.data == targets.data).item()

                all_labels.extend(targets.cpu().detach().numpy())
                all_preds.extend(preds.cpu().detach().numpy())

        valid_loss = valid_loss/len(valid_loader.dataset)
        valid_acc = correct/len(valid_loader.dataset)
        valid_metrics.step(all_labels, all_preds)

    print(('%13.4g' + '%13.4g'*3) % (train_loss, valid_loss, train_acc, valid_acc))
    return (
        valid_loss, valid_acc,
        valid_metrics.last_step_metrics(),
    )

def valid_adv_epoch(
        device, model_robust, model_teacher,
        valid_loader, valid_metrics, criterion,
        train_loss, train_acc, attacker, cfg
    ):
    #validate-the-model
    criterion_ori = nn.CrossEntropyLoss()
    with tqdm(enumerate(valid_loader), total = len(valid_loader)) as pbar:
        pbar.set_description(('%13s'  + '%13s' * 3) % ('Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'))
        with torch.no_grad():
            correct = 0
            valid_loss = 0
            all_labels = []
            all_preds = []
            model_teacher.eval()
            model_robust.eval()
            for batch_idx, (inputs, targets) in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                adv_inputs = attacker.perturb_PGD(inputs, targets)
                outputs = model_robust(adv_inputs)
                loss = criterion_ori(outputs, targets)
                valid_loss += loss.item() * inputs.size(0)
                ## calculate training metrics
                _, preds = torch.max(outputs.data, dim=-1)
                correct += torch.sum(preds.data == targets.data).item()

                all_labels.extend(targets.cpu().detach().numpy())
                all_preds.extend(preds.cpu().detach().numpy())

        valid_loss = valid_loss/len(valid_loader.dataset)
        valid_acc = correct/len(valid_loader.dataset)
        valid_metrics.step(all_labels, all_preds)

    print(('%13.4g' + '%13.4g'*3) % (train_loss, valid_loss, train_acc, valid_acc))
    return (
        valid_loss, valid_acc,
        valid_metrics.last_step_metrics(),
    )
