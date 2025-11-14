"""
Magic Georeferencer - AI-powered automatic image georeferencing for QGIS

This plugin uses the MatchAnything deep learning model to automatically
georeference ungeoreferenced images by matching them against basemap data.
"""

def classFactory(iface):
    """Load MagicGeoreferencer class from file magic_georeferencer.

    Args:
        iface: A QGIS interface instance.
    """
    from .magic_georeferencer import MagicGeoreferencer
    return MagicGeoreferencer(iface)
