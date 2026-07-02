"""
Logging utilities for Magic Georeferencer

Provides a single logger for the plugin that writes to Python logging and,
when running inside QGIS, mirrors warnings/errors to the QGIS message log.
"""

import logging

LOGGER_NAME = 'MagicGeoreferencer'

try:
    from qgis.core import Qgis, QgsMessageLog
    _QGIS_AVAILABLE = True
except ImportError:
    _QGIS_AVAILABLE = False


class _QgsMessageLogHandler(logging.Handler):
    """Forward log records to the QGIS message log panel."""

    def emit(self, record):
        try:
            if record.levelno >= logging.ERROR:
                level = Qgis.Critical
            elif record.levelno >= logging.WARNING:
                level = Qgis.Warning
            else:
                level = Qgis.Info
            QgsMessageLog.logMessage(self.format(record), LOGGER_NAME, level)
        except Exception:
            self.handleError(record)


def get_logger(name: str = None) -> logging.Logger:
    """Get the plugin logger (or a child logger for a module).

    Args:
        name: Optional child name, e.g. 'matcher' -> 'MagicGeoreferencer.matcher'

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(LOGGER_NAME)

    if not getattr(logger, '_magic_georeferencer_configured', False):
        logger.setLevel(logging.INFO)

        # Console handler (QGIS Python console / stdout)
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(name)s [%(levelname)s] %(message)s'))
        logger.addHandler(console)

        # QGIS message log for warnings and errors
        if _QGIS_AVAILABLE:
            qgis_handler = _QgsMessageLogHandler()
            qgis_handler.setLevel(logging.WARNING)
            qgis_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(qgis_handler)

        logger._magic_georeferencer_configured = True

    return logger.getChild(name) if name else logger


def set_debug(enabled: bool):
    """Toggle verbose debug logging for the whole plugin."""
    logging.getLogger(LOGGER_NAME).setLevel(logging.DEBUG if enabled else logging.INFO)
