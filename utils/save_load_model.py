import os
import torch

from utils.common import get_save_path, get_load_path

def save_model(model, optimizer, epoch, loss, best_loss, classifier_idx, fold_idx, path):
    save_path = get_save_path(classifier_idx, fold_idx, path)
    os.makedirs(save_path, exist_ok=True)
    
    # Save the last checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(save_path, 'last_checkpoint.pth'))
    
    # Save the best model based on loss
    if best_loss is not None:
        if loss < best_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(save_path, 'best_model.pth'))

def load_model(model, optimizer, classifier_idx, fold_idx, path):
    load_path = get_load_path(classifier_idx, fold_idx, path)
    last_checkpoint = os.path.join(load_path, 'last_checkpoint.pth')
    best_model = os.path.join(load_path, 'best_model.pth')
    
    if os.path.exists(best_model):
        checkpoint = torch.load(best_model)
    elif os.path.exists(last_checkpoint):
        checkpoint = torch.load(last_checkpoint)
    else:
        print("No checkpoint found!")
        return model, optimizer, 0, 0, 0
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss