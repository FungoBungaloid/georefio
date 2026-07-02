"""
UI components for Magic Georeferencer

Dialogs are imported lazily so opening one dialog doesn't pull in the
dependencies of all of them.
"""

_LAZY_IMPORTS = {
    'MagicGeoreferencerDialog': ('.main_dialog', 'MagicGeoreferencerDialog'),
    'ProgressDialog': ('.progress_dialog', 'ProgressDialog'),
    'ConfidenceViewer': ('.confidence_viewer', 'ConfidenceViewer'),
    'SettingsDialog': ('.settings_dialog', 'SettingsDialog'),
    'BatchDialog': ('.batch_dialog', 'BatchDialog'),
    'AlignmentDialog': ('.alignment_dialog', 'AlignmentDialog'),
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
