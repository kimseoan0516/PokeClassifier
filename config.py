from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    name: str
    backbone: str           # resnet18, resnet50, efficientnet_b0
    pretrained: bool
    fine_tune_mode: str     # scratch, frozen, partial, full
    num_epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_classes: int = 150
    image_size: int = 224
    num_workers: int = 2
    seed: int = 42


EXPERIMENTS = [
    ExperimentConfig(
        name="exp1_resnet18_scratch",
        backbone="resnet18",
        pretrained=False,
        fine_tune_mode="scratch",
        num_epochs=5,
        lr=1e-3,
    ),
    ExperimentConfig(
        name="exp2_resnet18_frozen",
        backbone="resnet18",
        pretrained=True,
        fine_tune_mode="frozen",
        num_epochs=5,
        lr=1e-3,
    ),
    ExperimentConfig(
        name="exp3_resnet50_partial",
        backbone="resnet50",
        pretrained=True,
        fine_tune_mode="partial",
        num_epochs=5,
        lr=5e-4,
    ),
    ExperimentConfig(
        name="exp4_efficientnet_full",
        backbone="efficientnet_b0",
        pretrained=True,
        fine_tune_mode="full",
        num_epochs=5,
        lr=1e-4,
    ),
]
