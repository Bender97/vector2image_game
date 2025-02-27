import torch
from torcheval.metrics import BinaryAUROC, BinaryConfusionMatrix

from prettytable import PrettyTable
from termcolor import colored


def get_CM(pred, gt):
    """ compute the Confusion Matrix between prediction and ground truth binary labels """
    pred_cat = torch.Tensor(pred).long()
    gt_cat   = torch.Tensor(gt).long()
    metric = BinaryConfusionMatrix()
    metric.update(pred_cat, gt_cat)
    cm = metric.compute().numpy()
    return cm

def get_AUROC(pred, gt):
    """ compute the Area Under ROC between prediction and ground truth binary labels """
    pred_cat = torch.Tensor(pred)
    gt_cat   = torch.Tensor(gt)
    metric = BinaryAUROC()
    metric.update(pred_cat, gt_cat)
    cm = metric.compute()
    return cm

def get_IoU(cm, pos=True):
    """ compute the Intersection over Union between prediction and ground truth binary labels """
    tn = cm[0, 0]
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    if pos:
        if (tp + fn + fp)==0:
            return 0
        return tp / (tp + fn + fp)
    if (tn + fn + fp)==0:
        return 0
    return tn / (tn + fn + fp)

def compute_metrics(pred, gt):
    cm   = get_CM(pred, gt)
    ioup = get_IoU(cm)
    ioun = get_IoU(cm, pos=False)
    iouavg = (ioup + ioun) / 2.0
    acc  = cm.trace() / cm.sum()
    auc = get_AUROC(pred, gt)
    return {"IoU+": ioup, "IoU-": ioun, "IoUavg": iouavg, "Acc": acc, "AUC": auc, "cm": cm, "loss": 1e9}

def nice_print(metrics):
    
    table = PrettyTable()
    table.field_names = ["Metric", "IoU+", "IoU-", "IoUavg", "Accuracy", "AUC", "Avg Loss"]
    
    table.add_row([colored("Value", "green"), 
                   f"{metrics['IoU+']*100:.2f}%", 
                   f"{metrics['IoU-']*100:.2f}%", 
                   f"{metrics['IoUavg']*100:.2f}%", 
                   f"{metrics['Acc']*100:.2f}%", 
                   f"{metrics['AUC']*100:.2f}%",
                   f"{metrics['loss']:.4f}"])
    
    print(table)
    if 'cm' in metrics:
        cm = metrics['cm']
        table = PrettyTable()
        table.field_names = ["Confusion Matrix:", "Predicted Negative", "Predicted Positive"]
        table.add_row(["Actual Negative", cm[0, 0], cm[0, 1]])
        table.add_row(["Actual Positive", cm[1, 0], cm[1, 1]])
        print(table)

def compute_avg_metrics(results):
    avg_metrics = {"IoU+": 0, "IoU-": 0, "IoUavg": 0, "Acc": 0, "AUC": 0, "loss": 0}
    for res in results:
        if "best_metrics_valid" in res:
            items = res["best_metrics_valid"].items()
        else:
            items = res.items()
        for key, value in items:
            if key!='cm':
                avg_metrics[key] += value
    for key in avg_metrics.keys():
        avg_metrics[key] /= len(results)
    return avg_metrics

def compute_best_classifier(avg_results):
    best_classifiers = {"IoU+": None, "IoU-": None, "IoUavg": None, "Acc": None, "AUC": None, "loss": None}
    best_values      = {"IoU+": 0, "IoU-": 0, "IoUavg": 0, "Acc": 0, "AUC": 0, "loss": 1e9}

    for classifier_idx, metrics in enumerate(avg_results):
        for key in best_values.keys():
            if key=="loss":
                if metrics[key] < best_values[key]:
                    best_values[key] = metrics[key]
                    best_classifiers[key] = classifier_idx
            elif metrics[key] > best_values[key]:
                best_values[key] = metrics[key]
                best_classifiers[key] = classifier_idx
    return best_classifiers, best_values