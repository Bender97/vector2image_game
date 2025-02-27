import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
device = "cuda" if torch.cuda.is_available() else "cpu"


weights = ResNet18_Weights.DEFAULT

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=weights, progress=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_pred = self.resnet(x)
        return y_pred
    
def create_model(ilr, scheduler_rate, lr_last_fc_weight):
    model = Model()

    """ change the last layer to have 2 channels
        - resnet18 last layer has  512 feats
        - resnet34 last layer has 1024 feats
        - resnet50 last layer has 2048 feats
    """
    model.resnet.fc = nn.Sequential(
        nn.Linear(512, 2),
    )

    model = model.to(device)
    model.train();

    lr = ilr

    optimizer = optim.SGD([
        {'params': model.resnet.fc.parameters(), 'lr': lr_last_fc_weight * lr},  # 20x learning rate for FC layer
        {'params': [param for name, param in model.named_parameters() if "fc" not in name], 'lr': lr}  # Base LR for rest
    ])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_rate)

    return model, optimizer, scheduler