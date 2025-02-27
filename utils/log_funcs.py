import os
from utils.common import get_save_path, get_load_path
from prettytable import PrettyTable # pip install prettytable

def LOG(mode, metrics, epoch, classifier_idx, fold_idx, path):
    save_path = os.path.join(get_save_path(classifier_idx, fold_idx, path), "log.txt")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a") as f:
        f.write(f"Epoch {epoch}, mode {mode} :\n")
        for key, value in metrics.items():
            if key != "cm":
                f.write(f"{key}: {value}\n")
            f.write("\n")
        f.write("\n")
def LOG_best(loss, classifier_idx, fold_idx, path):
    save_path = os.path.join(get_save_path(classifier_idx, fold_idx, path), "log.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a") as f:
        f.write("BEST MODEL FOUND\n")
        f.write(f"New best Loss is {loss}\n")
        f.write("\n")

def LOG_norm(min_, max_, classifier_idx, fold_idx, path):
    save_path = os.path.join(get_save_path(classifier_idx, fold_idx, path), "log.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a") as f:
        f.write("Normalization parameters:\n")
        f.write(f"Min: {min_}\n")
        f.write(f"Max: {max_}\n")
        f.write("\n")

def LOG_loss_weights(num_0, num_1, w_0, w_1, classifier_idx, fold_idx, path):
    save_path = os.path.join(get_save_path(classifier_idx, fold_idx, path), "log.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a") as f:
        f.write("Absoulte number of samples:\n")
        f.write(f" - Negative: {num_0}\n")
        f.write(f" - Positive: {num_1}\n")
        f.write("Loss weights:\n")
        f.write(f" - Negative: {w_0}\n")
        f.write(f" - Positive: {w_1}\n")
        f.write("\n")

def LOG_training_hyperparams(lr, scheduler_rate, lr_last_fc_weight, train_batch_size, classifier_idx, fold_idx, path):
    save_path = os.path.join(get_save_path(classifier_idx, fold_idx, path), "log.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a") as f:
        f.write("Training hyperparameters:\n")
        f.write(f" - Batch size: {train_batch_size}\n")
        f.write(f" - Learning rate: {lr}\n")
        f.write(f" - Scheduler rate: {scheduler_rate}\n")
        f.write(f" - Last FC layer weight: {lr_last_fc_weight}\n")
        f.write("\n")

def LOG_metrics(metrics, msg, classifier_idx, fold_idx, path):
    save_path = os.path.join(get_save_path(classifier_idx, fold_idx, path), "log.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a") as f:
        if msg is not None and type(msg) == str:
            f.write(msg + "\n")
        else:
            print("warning: msg is not a string")
            
        table = PrettyTable()
        table.field_names = ["Metric", "IoU+", "IoU-", "IoUavg", "Accuracy", "AUC", "Avg Loss"]
        
        table.add_row(["Value", 
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