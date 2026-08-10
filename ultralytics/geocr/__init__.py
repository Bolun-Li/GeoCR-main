"""GeoCR core modules.

The package exposes the three components described by the project:
Cross-Modal Prior Extraction (CPE), Frequency-Decoupled Feature
Calibration (FFC), and Geometric Difficulty Rebalancing (GDR).
"""

from .cpe import CrossModalPriorStore, GatedPromptFusion
from .ffc import DEFAULT_FFC_CONFIG, FrequencyDecoupledFeatureCalibration
from .gdr import GeometricDifficultyRebalancing
from .gdr_curriculum import (
    BackgroundEnvironmentPrototypeBank,
    ClassSimilarityGraph,
    EnvironmentGuidedCurriculum,
    ForegroundPrototypeBank,
    GDRCurriculumConfig,
    PrototypeBank,
)

__all__ = (
    "DEFAULT_FFC_CONFIG",
    "BackgroundEnvironmentPrototypeBank",
    "ClassSimilarityGraph",
    "CrossModalPriorStore",
    "EnvironmentGuidedCurriculum",
    "ForegroundPrototypeBank",
    "FrequencyDecoupledFeatureCalibration",
    "GDRCurriculumConfig",
    "GatedPromptFusion",
    "GeometricDifficultyRebalancing",
    "PrototypeBank",
)
