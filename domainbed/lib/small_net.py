import torch
import timm
from domainbed.lib.vision_transformer import Identity


class MobileNetV3(torch.nn.Module):
    KNOWN_MODELS = {
        'MobileNetV3': timm.models.mobilenetv3.mobilenetv3_small_100
    }
    def __init__(self, input_shape, hparams):
        super().__init__()
        func = self.KNOWN_MODELS[hparams['backbone']]
        self.network = func(pretrained=True)
        self.n_outputs = 1000
        # self.network.head = Identity()
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)
    

class EfficientNetV2(torch.nn.Module):
    KNOWN_MODELS = {
        'EfficientNetV2': timm.models.efficientnet.efficientnet_b3
    }
    def __init__(self, input_shape, hparams):
        super().__init__()
        func = self.KNOWN_MODELS[hparams['backbone']]
        self.network = func(pretrained=True)
        # self.n_outputs = self.network.norm.normalized_shape[0]
        self.n_outputs = 1000
        self.network.head = Identity()
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)
    
class ConvNext(torch.nn.Module):
    KNOWN_MODELS = {
        'ConvNext': timm.models.convnext.convnext_base_in22k
    }
    def __init__(self, input_shape, hparams):
        super().__init__()
        func = self.KNOWN_MODELS[hparams['backbone']]
        self.network = func(pretrained=True)
        # self.n_outputs = self.network.norm_pre.norm.normalized_shape[0]
        self.n_outputs = 21841
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)
    
class RegNetY(torch.nn.Module):
    KNOWN_MODELS = {
        'RegNetY': timm.models.regnet.regnety_160,
    }
    def __init__(self, input_shape, hparams):
        super().__init__()
        func = self.KNOWN_MODELS[hparams['backbone']]
        self.network = func(pretrained=True)
        # self.n_outputs = self.network.norm_pre.norm.normalized_shape[0]
        self.network.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(1),
        )
        self.n_outputs = 3024
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)