import torch
import torch.nn as nn
from torchvision import models
from config import ExperimentConfig


def build_model(cfg: ExperimentConfig) -> nn.Module:
    backbone = cfg.backbone
    pretrained = cfg.pretrained
    mode = cfg.fine_tune_mode
    num_classes = cfg.num_classes

    weights_arg = "DEFAULT" if pretrained else None

    if backbone == "resnet18":
        model = models.resnet18(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        _apply_fine_tune_mode(model, mode, head_name="fc")

    elif backbone == "resnet50":
        model = models.resnet50(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        _apply_fine_tune_mode(model, mode, head_name="fc")

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        _apply_fine_tune_mode(model, mode, head_name="classifier")

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model


def _apply_fine_tune_mode(model: nn.Module, mode: str, head_name: str):
    if mode == "scratch":
        # All parameters trainable (already the default)
        pass

    elif mode == "frozen":
        # Freeze everything except the classification head
        for name, param in model.named_parameters():
            if not name.startswith(head_name):
                param.requires_grad = False

    elif mode == "partial":
        # Freeze early layers; unfreeze the last layer group + head
        # Works for ResNet: freeze layer1/layer2, unfreeze layer3/layer4/fc
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in ["layer3", "layer4", head_name,
                                                 "features.6", "features.7", "features.8",
                                                 "classifier"]):
                param.requires_grad = True

    elif mode == "full":
        # All parameters trainable
        pass

    else:
        raise ValueError(f"Unknown fine_tune_mode: {mode}")


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
