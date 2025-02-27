from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from utils.metrics import compute_metrics, nice_print
from utils.log_funcs import *
from utils.save_load_model import save_model, load_model

device = "cuda" if torch.cuda.is_available() else "cpu"
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def train_classifier(model, optimizer, scheduler, loss_f, normalize_transform, trainloader, validloader, max_epochs, classifier_idx, fold_idx, path, debug=False):
    global writer
    
    if fold_idx is not None:
        print("Training classifier", classifier_idx, "fold", fold_idx)
    else:
        print("Training classifier", classifier_idx, "on the whole dataset")
    
    train_step = val_step = 0

    ## initialize the best loss to a very high value
    best_loss_train = best_loss_valid = 1e9
    best_metrics_train = {"IoU+": 0, "IoU-": 0, "IoUavg": 0, "Acc": 0, "AUC": 0, "loss": 1e9}
    best_metrics_valid = {"IoU+": 0, "IoU-": 0, "IoUavg": 0, "Acc": 0, "AUC": 0, "loss": 1e9}

    for epoch in range(max_epochs):
        model.train()
        trainlosses, validlosses = [], []
        pred, gt = [], []

        pbar = tqdm(trainloader)
        pbar.set_postfix(avg_train_loss=np.inf)

        for data, gtlabel in pbar:
            optimizer.zero_grad()
            data = normalize_transform(data.permute(0, 3, 1, 2))
            data, gtlabel = data.to(device).float(), gtlabel.to(device).float()
            res  = model(data)
            pred += (F.softmax(res, dim=1).argmax(dim=1).detach().cpu().numpy()).astype(int).tolist()
            gt   +=   gtlabel.detach().cpu().numpy().astype(int).tolist()

            loss = loss_f(res, gtlabel.long())
            
            loss.backward()
            optimizer.step()
            trainlosses.append(loss.item())
            writer.add_scalar("Loss/train", loss.item(), train_step)
            train_step += 1
            pbar.set_postfix(avg_train_loss=f"{np.array(trainlosses).mean():.4f}")
            if debug:
                break
            
        train_loss = np.array(trainlosses).mean()
        print(f"epoch [{str(epoch):3s}] - train")
        
        res = compute_metrics(pred, gt)
        res["loss"] = train_loss
        nice_print(res)

        LOG("train", res, epoch, classifier_idx, fold_idx, path=path)

        save_model(model, optimizer, epoch, train_loss, best_loss=None, classifier_idx=classifier_idx, fold_idx=fold_idx, path=path)
        if train_loss<best_loss_train:
            best_loss_train = train_loss
            best_metrics_train = res

        model.eval()
        with torch.no_grad():
            pred, gt = [], []
            pbar = tqdm(validloader); pbar.set_postfix(avg_valid_loss=np.inf)
            
            for data, gtlabel in pbar:
                data = normalize_transform(data.permute(0, 3, 1, 2))
                
                data, gtlabel = data.to(device).float(), gtlabel.to(device).float()
                res  = model(data)

                loss = loss_f(res, gtlabel.long())

                validlosses.append(loss.item())
                writer.add_scalar("Loss/valid", loss.item(), val_step)
                val_step += 1
                
                pred.append(torch.nn.functional.softmax(res, dim=-1).argmax(dim=-1).item())
                gt.append(gtlabel.item())
                pbar.set_postfix(avg_valid_loss=f"{np.array(validlosses).mean():.4f}")
                if debug:
                    break
            
            valid_loss = np.array(validlosses).mean()
            print(f"epoch [{str(epoch):3s}] -  valid")
            res = compute_metrics(pred, gt)
            res["loss"] = valid_loss
            nice_print(res)
            LOG("valid", res, epoch, classifier_idx, fold_idx, path=path)

            save_model(model, optimizer, epoch, valid_loss, best_loss_valid, classifier_idx, fold_idx, path)
            if valid_loss<best_loss_valid:
                best_loss_valid = valid_loss
                best_metrics_valid = res
                LOG_best(valid_loss, classifier_idx, fold_idx, path=path)
            
        scheduler.step()
    return {"best_metrics_train": best_metrics_train, "best_metrics_valid": best_metrics_valid}
    
def test_classifier(model, loss_f, testloader, normalize_transform):
    model.eval()
    testlosses = []
    test_step = 0
    with torch.no_grad():
        pred, gt = [], []
        logits = np.zeros((len(testloader), 2))
        pbar = tqdm(testloader); pbar.set_postfix(avg_test_loss=np.inf)
        
        for data, gtlabel in pbar:
            data = normalize_transform(data.permute(0, 3, 1, 2))
            
            data, gtlabel = data.to(device).float(), gtlabel.to(device).float()
            res  = model(data)
            loss = loss_f(res, gtlabel.long())
            testlosses.append(loss.item())
            logits[test_step] = res.cpu().numpy()
            writer.add_scalar("Loss/test", loss.item(), test_step)
            test_step += 1
            
            pred.append(torch.nn.functional.softmax(res, dim=-1).argmax(dim=-1).item())
            gt.append(gtlabel.item())
            pbar.set_postfix(avg_test_loss=f"{np.array(testlosses).mean():.4f}")
        
    test_loss = np.array(testlosses).mean()
    res = compute_metrics(pred, gt)
    res["loss"] = test_loss
    return res, logits