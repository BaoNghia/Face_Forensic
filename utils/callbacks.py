import torch
import logging
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience = 7, mode = 'min', delta = 0, path='./Checkpoint.pt', verbose = False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            mode (str): condition to save checkpoint.
                            Default: "min"
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False          
        """
        assert mode in ['min', 'max']
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.verbose = verbose
        self.delta = delta
        self.path = path
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
            self.delta *= -1
        else:
            self.monitor_op = np.greater
            self.best = -np.Inf
            self.delta *= 1

    def __call__(self, current, checkpoint):
        if self.monitor_op(current -  self.delta, self.best):
            self.best = current
            self.save_checkpoint(current, checkpoint)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


    def save_checkpoint(self, current, checkpoint):
        '''Saves model when validation loss decrease.'''
        logging.info("====> Save best epoch")
        torch.save(checkpoint, self.path)