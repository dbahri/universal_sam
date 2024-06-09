from utils.register import Registry 
from utils.configurable import configurable
import torch.nn as nn
import torch.nn.functional as F
import torch

MODELS_REGISTRY = Registry("Models")


def _cfg_to_simple(args):
    return {
        "num_classes": args.n_classes,
    }


class SimpleNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        output = F.log_softmax(x, dim=1)
        return output



@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_simple)
def simple_mnist(num_classes=10):
    return SimpleNet(input_dim=28*28, num_classes=num_classes)


def build_model(args):
    model = MODELS_REGISTRY.get(args.model)(args)
    model = model.cuda()
    return model