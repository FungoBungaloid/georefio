"""
Core components for Magic Georeferencer

Submodules are imported lazily so that lightweight components (e.g. the
vision API client or location providers) can be used without pulling in
heavy dependencies like torch/cv2, and so plugin startup stays fast.
"""

_LAZY_IMPORTS = {
    'ModelManager': ('.model_manager', 'ModelManager'),
    'Matcher': ('.matcher', 'Matcher'),
    'MatchResult': ('.matcher', 'MatchResult'),
    'TileFetcher': ('.tile_fetcher', 'TileFetcher'),
    'GCPGenerator': ('.gcp_generator', 'GCPGenerator'),
    'Georeferencer': ('.georeferencer', 'Georeferencer'),
    'VisionAPIClient': ('.vision_api', 'VisionAPIClient'),
    'APIProvider': ('.vision_api', 'APIProvider'),
    'BoundingBoxEstimate': ('.vision_api', 'BoundingBoxEstimate'),
    'VISION_MODELS': ('.vision_api', 'VISION_MODELS'),
    'estimate_batch_cost': ('.vision_api', 'estimate_batch_cost'),
    'BatchProcessor': ('.batch_processor', 'BatchProcessor'),
    'BatchConfig': ('.batch_processor', 'BatchConfig'),
    'BatchProgress': ('.batch_processor', 'BatchProgress'),
    'BatchItemResult': ('.batch_processor', 'BatchItemResult'),
    'BatchItemStatus': ('.batch_processor', 'BatchItemStatus'),
    'find_images_in_path': ('.batch_processor', 'find_images_in_path'),
    'build_hint_chain': ('.location_providers', 'build_hint_chain'),
    'LocationHintProvider': ('.location_providers', 'LocationHintProvider'),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module_name, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
