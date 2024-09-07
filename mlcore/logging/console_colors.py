#! /usr/bin/env python3
"""
Author: Zooey He
Email: zhuohonghe@gmail.com
Date: 2024-07-04
"""

import re
import string

"""
Methods for coloring text on the console / command line / terminal.
"""

class BCOLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    MAGENTA = '\033[0;35m'
    BWHITE = '\033[0;97m'


def remove_color_escapes(text: str) -> str:
    """
    Removes coloring escape characters (defined in BCOLORS) from a string.
    """
    s = ''.join([c for c in text if c in string.printable])  # Filter out non-printable chars
    s = re.sub('\[[0-9]{1,2}m', '', s)  # Substitute out color escapes
    s = re.sub('\[[0-9]{1};[0-9]{2}m', '', s)
    return s


def color_log_line(line: str) -> str:
    """
    Given a logging line. Return the line with cmd coloration.
    """
    line = line.replace("DEBUG:", blue("DEBUG:"))
    line = line.replace("INFO:", green("INFO:"))
    line = line.replace("WARNING:", yellow("WARNING:"))
    line = line.replace("WARN:", yellow("WARN:"))
    line = line.replace("ERROR:", red("ERROR:"))
    line = line.replace("CRITICAL:", red("CRITICAL:"))

    header_template = r'(==== .* ====)'
    line = re.sub(header_template, '\n' + red(r'\1'), line)

    return line


def bold(string: str):
    return f'{BCOLORS.BOLD}{string}{BCOLORS.ENDC}'


def uline(string: str):
    return f'{BCOLORS.UNDERLINE}{string}{BCOLORS.ENDC}'


def cyan(string: str):
    return f'{BCOLORS.OKCYAN}{string}{BCOLORS.ENDC}'


def blue(string: str):
    return f'{BCOLORS.OKBLUE}{string}{BCOLORS.ENDC}'


def magenta(string: str):
    return f'{BCOLORS.HEADER}{string}{BCOLORS.ENDC}'


def yellow(string: str):
    return f'{BCOLORS.WARNING}{string}{BCOLORS.ENDC}'


def green(string: str):
    return f'{BCOLORS.OKGREEN}{string}{BCOLORS.ENDC}'


def red(string: str):
    return f'{BCOLORS.FAIL}{string}{BCOLORS.ENDC}'


def magentab(string: str):
    return f'{BCOLORS.MAGENTA}{string}{BCOLORS.ENDC}'


def bwhite(string: str):
    return f'{BCOLORS.BWHITE}{string}{BCOLORS.ENDC}'