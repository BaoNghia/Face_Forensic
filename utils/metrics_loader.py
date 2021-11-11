from sklearn import metrics as skmetrics
from utils import metrics as cmetrics
import numpy as np

# You can use this file with your custom metrics
class Metrics:
    def __init__(self, metric_names):
        self.metric_names = metric_names
        # initialize a metric dictionary
        self.metric_dict = {metric_name: [] for metric_name in self.metric_names}

    def reset(self):
        self.metric_dict = {metric_name: [] for metric_name in self.metric_names}
 
    def step(self, labels, preds):
        for metric in self.metric_names:
            # get the metric function
            try:
                do_metric = getattr(skmetrics, metric)
            except:
                do_metric = getattr(cmetrics, metric, "The metric {} is not implemented".format(metric))

            # check if metric require average method
            try:
                self.metric_dict[metric].append(do_metric(labels, preds, average="macro"))
            except:
                self.metric_dict[metric].append(do_metric(labels, preds))
          
    def epoch(self):
        # calculate metrics for an entire epoch
        avg = [sum(metric) / (len(metric)) for metric in self.metric_dict.values()]
        metric_as_dict = dict(zip(self.metric_names, avg))
        # reset metric_dict for new epoch
        self.reset()
        return metric_as_dict

    def last_step_metrics(self):
        # return metrics of last steps
        values = [self.metric_dict[metric][-1] for metric in self.metric_names]
        metric_as_dict = dict(zip(self.metric_names, values))
        self.reset()
        return metric_as_dict