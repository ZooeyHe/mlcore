"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2024-07-04
"""

import logging

import mlcore.logging.console_colors as cc

"""
Implementation of colored formatting.
"""

class ConsoleColorFormatter(logging.Formatter):
    """
    Custom formatter with coloring for the console / terminal.
    """
    def format(self, record):
        """
        Overrides the existing format function. Adds automatic coloring to log line.
        """
        line = super().format(record)
        line = cc.color_log_line(line)
        return line


class FileDecoloringFormatter(logging.Formatter):
    """
    Custom formatter which removes coloring form input string (for file).
    """
    def format(self, record):
        """
        Overrides existing format function. Removes colors escape characters from line.
        """
        line = super().format(record)
        line = cc.remove_color_escapes(line)
        return line