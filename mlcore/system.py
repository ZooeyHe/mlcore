"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2024-08-31
"""


import os


"""
System Utilities.
"""


def display_detected() -> bool:
    """
    Determine if a display is detected from this process.
    """
    return bool( "DISPLAY" in os.environ )