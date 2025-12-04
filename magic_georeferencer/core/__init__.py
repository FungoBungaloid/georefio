"""
Core components for Magic Georeferencer
"""

from .model_manager import ModelManager
from .matcher import Matcher, MatchResult
from .tile_fetcher import TileFetcher
from .gcp_generator import GCPGenerator
from .georeferencer import Georeferencer
from .vision_api import (
    VisionAPIClient,
    APIProvider,
    BoundingBoxEstimate,
    VISION_MODELS,
    estimate_batch_cost
)
from .batch_processor import (
    BatchProcessor,
    BatchConfig,
    BatchProgress,
    BatchItemResult,
    BatchItemStatus,
    find_images_in_path
)

__all__ = [
    'ModelManager',
    'Matcher',
    'MatchResult',
    'TileFetcher',
    'GCPGenerator',
    'Georeferencer',
    'VisionAPIClient',
    'APIProvider',
    'BoundingBoxEstimate',
    'VISION_MODELS',
    'estimate_batch_cost',
    'BatchProcessor',
    'BatchConfig',
    'BatchProgress',
    'BatchItemResult',
    'BatchItemStatus',
    'find_images_in_path'
]
