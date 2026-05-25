"""Machine-learning pipeline helpers for echopress."""

from .dataset import build_dataset, DatasetBuildConfig
from .splits import build_split, SplitConfig
from .preprocess import fit_preprocessing, transform_with_preprocessing, PreprocessConfig
from .models import build_model, ModelConfig
from .train import train_model, TrainConfig
from .evaluate import evaluate_model, EvaluateConfig

__all__ = [
    "DatasetBuildConfig",
    "SplitConfig",
    "PreprocessConfig",
    "ModelConfig",
    "TrainConfig",
    "EvaluateConfig",
    "build_dataset",
    "build_split",
    "fit_preprocessing",
    "transform_with_preprocessing",
    "build_model",
    "train_model",
    "evaluate_model",
]
