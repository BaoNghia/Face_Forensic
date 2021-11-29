import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

      
def train_epoch(
        epoch, num_epochs,
        device, adversarial_module,
        train_loader, train_metrics,
        criterion, optimizer
    ):
    model_robust = adversarial_module.model_robust
    model_natural = adversarial_module.model_natural
    # training-the-model
    with tqdm(enumerate(train_loader), total = len(train_loader)) as pbar:
        train_loss = 0
        mloss = 0
        correct = 0
        for batch_idx, (inputs, targets) in pbar:
            # move-tensors-to-GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            # # generate adversarial_sample
            optimizer.zero_grad()
            x_adv = adversarial_module(inputs)
            # zero the gradient beforehand
            model_robust.train()
            model_natural.train()
            optimizer.zero_grad()
            out_adv = model_robust(x_adv)
            out_natural = model_robust(inputs)
            out_orig = model_natural(inputs)
            # forward model and compute loss
            loss = criterion(out_adv, out_natural, out_orig, targets)
                  
            loss.backward()
            optimizer.step()
            # update-training-loss
            train_loss += loss.item()
            ## calculate training metrics
            outputs = model_robust(inputs)
            outputs_softmax = torch.softmax(outputs, dim=-1)
            probs, preds = torch.max(outputs_softmax.data, dim=-1)
            correct += torch.sum(preds.data == targets.data).item()
            train_metrics.step(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())

            ## pbar
            mem = '%.3g GB' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            mloss = (mloss * batch_idx + loss.item())/(batch_idx + 1)
            s = ('%13s' * 2 + '%13.4g' * 1) % ('%g/%g' % (epoch, num_epochs - 1), mem, mloss)
            pbar.set_description(s)
            pbar.set_postfix(lr = optimizer.param_groups[0]['lr'])

        train_loss = train_loss/len(train_loader.dataset)
        train_acc = correct/len(train_loader.dataset)

    return train_loss, train_acc, train_metrics.epoch()
        
    
def valid_epoch(
        device, adversarial_module,
        valid_loader, valid_metrics,
        criterion, train_loss, train_acc,
    ):
    model_robust = adversarial_module.model_robust
    model_natural = adversarial_module.model_natural
    #validate-the-model
    with tqdm(enumerate(valid_loader), total = len(valid_loader)) as pbar:
        pbar.set_description(('%13s'  + '%13s' * 3) % ('Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'))
        with torch.no_grad():
            correct = 0
            valid_loss = 0
            all_labels = []
            all_preds = []
            
            model_natural.eval()
            model_robust.eval()
            for batch_idx, (inputs, targets) in pbar:
                inputs = inputs.to(device)
                targets = targets.to(device)
                # generate adversarial_sample
                x_adv = adversarial_module(inputs)
                # forward model and compute loss
                out_adv = model_robust(x_adv)
                out_natural = model_robust(inputs)
                out_orig = model_natural(inputs)
                loss = criterion(out_adv, out_natural, out_orig, targets)
                valid_loss += loss.item()
                ## calculate training metrics
                outputs = model_robust(inputs)
                outputs_softmax = torch.softmax(outputs, dim=-1)
                probs, preds = torch.max(outputs_softmax.data, dim=-1)
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