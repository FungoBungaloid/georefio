"""
Bulk Georeferencer - Batch AI-powered automatic image georeferencing for QGIS

This plugin provides batch processing capabilities for georeferencing multiple
images using Vision AI for location estimation and the MatchAnything model
for feature matching.

Requires: Magic Georeferencer plugin
"""


def classFactory(iface):
    """Load BulkGeoreferencer class from file bulk_georeferencer.

    Args:
        iface: A QGIS interface instance.
    """
    from .bulk_georeferencer import BulkGeoreferencer
    return BulkGeoreferencer(iface)
