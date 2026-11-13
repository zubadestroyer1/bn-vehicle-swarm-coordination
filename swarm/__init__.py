"""City-scale autonomous vehicle swarm coordination via neurally updated Dynamic Bayesian Networks."""

from __future__ import annotations

from .config import (
    ControlConfig,
    DBNConfig,
    ExperimentConfig,
    MODEL_PRESETS,
    ModelPresetSpec,
    ObservationConfig,
    ScenarioConfig,
    TEST_SEEDS,
    TRAIN_SEEDS,
    TrainConfig,
    VAL_SEEDS,
)
from .contracts import (
    ABLATION_POLICY_NAMES,
    CORRIDOR_EDGE_FEATURE_DIM,
    CORRIDOR_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    MAIN_POLICY_NAMES,
    NEURAL_POLICY_NAMES,
    NODE_FEATURE_DIM,
    POLICY_NAMES,
    SCHEMA_VERSION,
)

__version__ = "0.2.0"

__all__ = [
    "ABLATION_POLICY_NAMES",
    "CORRIDOR_EDGE_FEATURE_DIM",
    "CORRIDOR_FEATURE_DIM",
    "ControlConfig",
    "DBNConfig",
    "EDGE_FEATURE_DIM",
    "ExperimentConfig",
    "MAIN_POLICY_NAMES",
    "MODEL_PRESETS",
    "ModelPresetSpec",
    "NEURAL_POLICY_NAMES",
    "NODE_FEATURE_DIM",
    "ObservationConfig",
    "POLICY_NAMES",
    "SCHEMA_VERSION",
    "ScenarioConfig",
    "TEST_SEEDS",
    "TRAIN_SEEDS",
    "TrainConfig",
    "VAL_SEEDS",
    "__version__",
]
